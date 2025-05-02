# Standard library imports
import os
import json
from datetime import datetime
import calendar

# Third-party imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch.nn.functional as F
from typing import List, Dict, Optional



class DateClassifier:
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
        # List of months in Russian
        months_in_russian = [
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ]
        dates = []
        for month in range(1, 13):  # Loop through months 1 to 12
            days_in_month = calendar.monthrange(2024, month)[1]  # Get the number of days in the month
            for day in range(1, days_in_month + 1):
                dates.append(f"{day} {months_in_russian[month - 1]}")
        return dates

    def replace_texts_with_numbers(self, text):
        # Инициализация моделей

        # Таблица для преобразования числительных
        NUMBERS = {
            "первого": 1, "второго": 2, "третьего": 3, "четвёртого": 4, "пятого": 5,
            "шестого": 6, "седьмого": 7, "восьмого": 8, "девятого": 9, "десятого": 10,
            "одиннадцатого": 11, "двенадцатого": 12, "тринадцатого": 13, "четырнадцатого": 14,
            "пятнадцатого": 15, "шестнадцатого": 16, "семнадцатого": 17, "восемнадцатого": 18,
            "девятнадцатого": 19, "двадцатого": 20, "двадцать первого": 21, "двадцать второго": 22,
            "двадцать третьего": 23, "двадцать четвёртого": 24, "двадцать пятого": 25,
            "двадцать шестого": 26, "двадцать седьмого": 27, "двадцать восьмого": 28,
            "двадцать девятого": 29, "тридцатого": 30, "тридцать первого": 31
        }

        tokens = text.split()
        new_tokens = []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            num_phrase = token

            # Проверяем составные числительные (два слова подряд)
            if i < len(tokens) - 1:
                num_phrase = f"{tokens[i]} {tokens[i + 1]}"
                if num_phrase in NUMBERS:
                    new_tokens.append(str(NUMBERS[num_phrase]))
                    i += 2
                    continue

            if token in NUMBERS:
                new_tokens.append(str(NUMBERS[token]))
            else:
                new_tokens.append(token)

            i += 1

        new_text = " ".join(new_tokens)
        return new_text

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

    def predict(self, query: str) -> Dict[str, Optional[datetime]]:
        self.model.eval()

        # 8️⃣ Inference - Using the Trained Model
        ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        def parse_dates(query: str):
            query = self.replace_texts_with_numbers(query)

            results = ner_pipeline(query)

            def datetime_from_text(text: str, end_of_day: bool) -> datetime:
                tokens = text.split()

                months_to_numbers = {
                    "января": 1,
                    "февраля": 2,
                    "марта": 3,
                    "апреля": 4,
                    "мая": 5,
                    "июня": 6,
                    "июля": 7,
                    "августа": 8,
                    "сентября": 9,
                    "октября": 10,
                    "ноября": 11,
                    "декабря": 12,
                }

                month_and_day_str = f"{months_to_numbers[tokens[1]]}-{tokens[0]}"

                year = datetime.now().year  # or set a specific year like 2025

                date = datetime.strptime(f"{year}-{month_and_day_str}", "%Y-%m-%d")

                if end_of_day:
                    date = date.replace(hour=23, minute=59)

                return date

            start_dates = [datetime_from_text(r["word"], False) for r in results if r["entity"] == "B-STARTDATE"]
            end_dates = [datetime_from_text(r["word"], True) for r in results if r["entity"] == "B-ENDDATE"]

            return {"B-STARTDATE": start_dates[0] if len(start_dates) > 0 else None, "B-ENDDATE": end_dates[0] if len(end_dates) > 0 else None}

        return parse_dates(query)


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