# Standard library imports
import os.path
import json
import re
from difflib import SequenceMatcher

# Third-party imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import pymorphy3

# Local imports
from city_db.city_db import CityDB
from typing import List, Dict



class RouteParser:
    def __init__(self):
        self.path: str = os.path.dirname(__file__)
        self.model: AutoModelForTokenClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.input_max_length = 32

        city_name_database = CityDB()
        city_name_database.load(filename='out\cities_db.json')
        self.city_names = city_name_database.get_cities('ru')

        self.morph = pymorphy3.MorphAnalyzer()

    def model_path(self):
        return os.path.join(self.path, 'model')

    def datasets_path(self):
        return os.path.join(self.path, 'datasets')

    def add_space_before_punctuation(self, text):
        return re.sub(r"(\S)([.,!?;:])", r"\1 \2", text)

    def normalize_city_name(self, city_name):
        words = city_name.split()

        parsed_first_word = self.morph.parse(words[0])[0]
        gender = parsed_first_word.tag.gender

        normalized_words = []
        for word in words:
            parsed = self.morph.parse(word)[0]

            if gender:
                normal_form = parsed.inflect({'nomn', gender})
            else:
                normal_form = parsed.inflect({'nomn'})

            if normal_form:
                normalized_words.append(normal_form.word)
            else:
                normalized_words.append(parsed.normal_form)

        normalized_city = " ".join(normalized_words).title()

        if normalized_city in self.city_names:
            return normalized_city, 1.0
        else:
            threshold = 0.95
            best_match = None
            best_ratio = 0

            for city in self.city_names:
                ratio = SequenceMatcher(None, city_name, city).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = city

            if best_match and best_ratio >= threshold:
                return best_match, best_ratio

        return city_name, 0.0

    def normalize_text_cities(self, text):
        words = text.split()
        normalized_words = []

        i = 0
        while i < len(words):
            found = False
            found_length = 0
            best_normalized_phrase = ""
            best_equality_ratio = 0.0
            for length in range(3, 0, -1):  # Проверяем фразы длиной от 3 до 1 слова
                phrase = " ".join(words[i:i + length])
                normalized_phrase, equality_ratio = self.normalize_city_name(phrase)
                if normalized_phrase in self.city_names and equality_ratio > best_equality_ratio:
                    best_normalized_phrase = normalized_phrase
                    best_equality_ratio = equality_ratio
                    found_length = length
                    found = True
            if found:
                i += found_length
                normalized_words.append(best_normalized_phrase)
            else:
                normalized_words.append(words[i])
                i += 1

        return " ".join(normalized_words)

    def learn(self) -> None:
        with open(os.path.join(self.datasets_path(), "dataset.json"), "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        for entry in dataset_json:
            for i, token in enumerate(entry["tokens"]):
                normalized_city_name, equality_ratio = self.normalize_city_name(token)
                if normalized_city_name in self.city_names:
                    entry["tokens"][i] = normalized_city_name
            print(entry["tokens"])

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(dataset_json)

        # 2️⃣ Load Pretrained BERT Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", model_max_length=self.input_max_length)

        new_tokens = self.city_names
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys())
        # add the tokens to the tokenizer vocabulary
        self.tokenizer.add_tokens(list(new_tokens))
        # add new, random embeddings for the new tokens

        # Define Label Mapping
        label_list = ["O", "B-DEP", "B-DEST"]
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}

        # 3️⃣ Tokenization Function with Proper Padding & Label Alignment
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                padding="max_length",  # ✅ Ensures uniform sequence lengths
                max_length=self.input_max_length,
                is_split_into_words=True
            )

            labels = []
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)  # Ignore padding tokens
                elif word_id != previous_word_id:
                    try:
                        labels.append(label2id[examples["labels"][word_id]])
                    except:
                        print(examples["tokens"])
                else:
                    labels.append(-100)  # Ignore subword tokens
                previous_word_id = word_id

            # ✅ Ensure labels match input length
            padding_length = len(tokenized_inputs["input_ids"]) - len(labels)
            labels += [-100] * padding_length

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        # Apply Tokenization
        tokenized_dataset = dataset.map(tokenize_and_align_labels)

        # 4️⃣ Load Pretrained Model for Token Classification
        self.model = AutoModelForTokenClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased", num_labels=len(label_list), id2label=id2label, label2id=label2id
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 5️⃣ Training Arguments
        training_args = TrainingArguments(
            output_dir="../../results",
            num_train_epochs=5,
            per_device_train_batch_size=2,
            logging_dir="./logs",
            logging_steps=1,
            save_strategy="no",
            per_device_eval_batch_size=4,
            dataloader_drop_last=True  # ✅ Prevents shape mismatches in batches
        )

        # 6️⃣ Data Collator for Padding
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        def focal_loss(logits, labels, gamma=2.0, reduction='mean'):
            """
            Computes Focal Loss, which down-weights easy examples and focuses on hard ones.

            Args:
                logits: Tensor of shape [batch_size, num_classes] - raw model outputs.
                targets: Tensor of shape [batch_size] - ground truth class indices.
                gamma: Focusing parameter; higher values put more weight on hard examples.
                reduction: 'mean' (default) returns mean loss, 'sum' sums all losses.

            Returns:
                Loss value (scalar).
            """
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
            prob = torch.exp(-ce_loss)
            focal_loss = (1 - prob) ** gamma * ce_loss

            return focal_loss.mean() if reduction == 'mean' else focal_loss.sum()

        # class CustomTrainer(Trainer):
        #     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        #         """
        #         Custom loss function supporting an optional num_items_in_batch parameter.
        #
        #         Args:
        #             model: The model to be trained.
        #             inputs: The input batch dictionary.
        #             return_outputs: Whether to return model outputs along with loss.
        #             num_items_in_batch (int, optional): Additional parameter, not used in loss computation.
        #
        #         Returns:
        #             loss: The computed loss.
        #             outputs (optional): The model outputs if return_outputs=True.
        #         """
        #         outputs = model(**inputs)
        #         logits = outputs.logits
        #         labels = inputs.get("labels")
        #
        #         # Example: Using Focal Loss
        #         ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
        #         gamma = 2.0  # Focusing parameter
        #         prob = torch.exp(-ce_loss)
        #         focal_loss = (1 - prob) ** gamma * ce_loss
        #         loss = focal_loss.mean()
        #
        #         return (loss, outputs) if return_outputs else loss

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs.get("labels")

                loss = focal_loss(logits, labels, gamma=2.0)  # Use focal loss
                # loss = label_smoothing_loss(logits, labels, smoothing=0.1)

                return (loss, outputs) if return_outputs else loss

        # 7️⃣ Train the Model
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,  # ✅ Ensures correct batch padding
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

    def save(self) -> None:
        # Save model and tokenizer
        self.model.save_pretrained(self.model_path())
        self.tokenizer.save_pretrained(self.model_path())

    def load(self) -> None:
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path(), model_max_length=self.input_max_length)

    def predict(self, text: str) -> Dict[str, List[str]]:
        self.model.eval()

        text = self.add_space_before_punctuation(text)
        text = self.normalize_text_cities(text)

        print(text)

        # 8️⃣ Inference - Using the Trained Model
        ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        def extract_locations(text):
            results = ner_pipeline(text)
            departure_points = [r["word"] for r in results if r["entity"] == "B-DEP"]
            destination_points = [r["word"] for r in results if r["entity"] == "B-DEST"]

            return {"B-DEP": departure_points, "B-DEST": destination_points}

        return extract_locations(text)


if __name__ == '__main__':
    dep_arr_classifier = RouteParser()
    dep_arr_classifier.learn()
    dep_arr_classifier.save()
    dep_arr_classifier.load()
    queries = [
        "хочу куда-то на выходных полететь из Дубая.",
        "Куда можно полететь из Вены на выходных плюс пятница?",
        "хочу в Манчестер",
        "Привет, как у тебя в доме?",
        "живу в Праге",
        "Хочу уехать на выходные, могу вылететь из Барселоны или Мадрида.",
        "Хочу куда-то полететь на выходных, могу вылетать из Праги, Брно или Вены.",
        "Я бы хотел полететь из Сургута в Якутск.",
        "Я бы хотел полететь в Сургут из Якутска.",
        "Я бы хотел полететь из Якутска в Сургут.",
        "Я бы хотел полететь в Якутск из Сургута.",
        "Я бы хотел полететь в Якутск из Сургута или Москвы.",
        "Я бы хотел полететь из Санкт-Петербурга в Якутск.",
        "Я бы хотел полететь из Новой Каховки в Якутск или Франкфурт-на-Майне.",
        "Я бы хотел полететь из Франкфурта-на-Майне в Новую Каховку.",
    ]
    for query in queries:
        dep_arr_data = dep_arr_classifier.predict(query)
        print(dep_arr_data)
