import os
import json
import sys
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification


class NNParserBase:
    def __init__(self, path: str):
        self.path: str = path
        self.model: AutoModelForTokenClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.input_max_length = 32

    def model_path(self):
        return os.path.join(self.path, 'model')

    def datasets_path(self):
        return os.path.join(self.path, 'datasets')

    def create_tokens(self):
        raise NotImplementedError  # Must be implemented in derived classes

    def learn(self) -> None:
        with open(os.path.join(self.datasets_path(), "dataset.json"), "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(dataset_json)

        # Load Pretrained BERT Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", model_max_length=self.input_max_length)

        new_tokens = self.create_tokens()
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys())
        self.tokenizer.add_tokens(list(new_tokens))

        # Define Label Mapping
        label_list = self.get_label_list()
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}

        # Tokenization Function with Padding & Label Alignment
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                padding="max_length",
                max_length=self.input_max_length,
                is_split_into_words=True
            )

            labels = []
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != previous_word_id:
                    try:
                        labels.append(label2id[examples["labels"][word_id]])
                    except KeyError as e:
                        print(f"Error: Key {e} not found in label2id. Tokens: {examples['tokens']}")
                        sys.exit(1)  # Stop the program with a non-zero exit code indicating an error
                else:
                    labels.append(-100)
                previous_word_id = word_id

            padding_length = len(tokenized_inputs["input_ids"]) - len(labels)
            labels += [-100] * padding_length

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        # Apply Tokenization
        tokenized_dataset = dataset.map(tokenize_and_align_labels)

        # Load Pretrained Model for Token Classification
        self.model = AutoModelForTokenClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased", num_labels=len(label_list), id2label=id2label, label2id=label2id
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Training Arguments
        training_args = TrainingArguments(
            output_dir="../../results",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            logging_dir="./logs",
            logging_steps=1,
            save_strategy="no",
            per_device_eval_batch_size=4,
            dataloader_drop_last=True
        )

        # Data Collator for Padding
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Train the Model
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

    def save(self) -> None:
        self.model.save_pretrained(self.model_path())
        self.tokenizer.save_pretrained(self.model_path())

    def load(self) -> None:
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path(), model_max_length=self.input_max_length)

    def get_label_list(self) -> List[str]:
        raise NotImplementedError  # Must be implemented in derived classes
