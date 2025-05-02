# Base class import
from NNParserBase.neural_parser_base import NeuralParserBase

# Standard library imports
import os
import re
import json
from typing import List, Dict
from difflib import SequenceMatcher

# Third-party imports
import torch
import pymorphy3
from transformers import pipeline

# Local imports
from city_db.city_db import CityDB
from route_parser.route_parser_trainer import RouteParserTrainer


class RouteParser(NeuralParserBase):
    def __init__(self):
        super().__init__(path=os.path.dirname(__file__), trainer_class=RouteParserTrainer)

        city_name_database = CityDB()
        city_name_database.load(filename='out\cities_db.json')
        self.city_names = city_name_database.get_cities('ru')

        self.morph = pymorphy3.MorphAnalyzer()

    def create_tokens(self):
        return self.city_names

    def get_label_list(self) -> List[str]:
        return ["O", "B-DEP", "B-DEST"]

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
        dataset_path = os.path.join(self.datasets_path(), "dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        # Preprocess: normalize tokens in dataset
        for entry in dataset_json:
            for i, token in enumerate(entry["tokens"]):
                normalized_city_name, _ = self.normalize_city_name(token)
                if normalized_city_name in self.city_names:
                    entry["tokens"][i] = normalized_city_name
            print(entry["tokens"])

        # Save preprocessed dataset back (or pass it directly if you modify NeuralParserBase to accept a dataset)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=2)

        super().learn()

    def predict(self, text: str) -> Dict[str, List[str]]:
        self.model.eval()

        text = self.add_space_before_punctuation(text)
        text = self.normalize_text_cities(text)

        print(text)

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
