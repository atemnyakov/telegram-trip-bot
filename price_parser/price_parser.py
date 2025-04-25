import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline, DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import Dataset
from typing import List, Dict
import torch
import torch.nn.functional as F
import json


class PriceParser:
    def __init__(self):
        self.path: str = os.path.dirname(__file__)
        self.model: AutoModelForTokenClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.input_max_length = 32

    def model_path(self):
        return os.path.join(self.path, 'model')

    def datasets_path(self):
        return os.path.join(self.path, 'datasets')

    def create_tokens(self):
        return []

    def learn(self) -> None:
        with open(os.path.join(self.datasets_path(), "dataset.json"), "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(dataset_json)

        # 2️⃣ Load Pretrained BERT Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", model_max_length=self.input_max_length)

        new_tokens = self.create_tokens()
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys())
        # add the tokens to the tokenizer vocabulary
        self.tokenizer.add_tokens(list(new_tokens))
        # add new, random embeddings for the new tokens

        # Define Label Mapping
        label_list = ["O", "B-STARTDATE", "B-ENDDATE"]
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
            per_device_train_batch_size=4,
            logging_dir="./logs",
            logging_steps=1,
            save_strategy="no",
            per_device_eval_batch_size=4,
            dataloader_drop_last=True  # ✅ Prevents shape mismatches in batches
        )

        # 6️⃣ Data Collator for Padding
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # 7️⃣ Train the Model
        trainer = Trainer(
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

        # 8️⃣ Inference - Using the Trained Model
        ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        def parse_price(text):
            text = self.replace_texts_with_numbers(text)
            print(text)
            results = ner_pipeline(text)
            start_dates = [r["word"] for r in results if r["entity"] == "B-STARTDATE"]
            end_dates = [r["word"] for r in results if r["entity"] == "B-ENDDATE"]

            return {"Start dates": start_dates, "End dates": end_dates}

        return parse_price(text)


if __name__ == '__main__':
    date_classifier = DateClassifier()
    date_classifier.learn()
    date_classifier.save()
    date_classifier.load()
    queries = [
        "Я хочу куда-то полететь с двадцать первого января по 22 января",
        "Я хочу куда-то полететь по двадцать второго января с 21 января",
        "В Берлине буду с 1 марта до 3 марта",
        "Буду в Берлине до 4 апреля"
    ]
    for query in queries:
        start_end_date = date_classifier.predict(query)
        print(start_end_date)