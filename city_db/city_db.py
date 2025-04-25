from typing import Optional, List

from langdetect import detect, DetectorFactory
import langid
from tqdm import tqdm  # Import tqdm for progress bar
import json
from lingua import LanguageDetectorBuilder, Language, IsoCode639_1, IsoCode639_3
import pycountry
from datetime import datetime
import re
import os


class CityDB:
    def __init__(self, path: str = os.path.dirname(__file__)):
        self.data = None
        self.path = path

    def create(self, filename: str = 'in\cities15000.txt'):
        def contains_cyrillic(text):
            return bool(re.search(r'[\u0400-\u04FF\u0500-\u052F]', text))

        with open(os.path.join(self.path, filename), "r", encoding="utf-8") as f:
            self.data = {}

            # Use tqdm to create a progress bar on the file read
            total_lines = sum(1 for line in f)  # Get the total number of lines for the progress bar
            f.seek(0)  # Reset the file pointer to the beginning after counting lines

            for line in tqdm(f, desc="Processing lines", total=total_lines, unit="line", ncols=100):  # Set total for progress bar
                line_split = line.split('\t')
                if len(line_split) >= 3:
                    city_id = line_split[0]
                    self.data[city_id] = {}
                    english_name = line_split[2]
                    self.data[city_id]['en'] = [english_name]
                    names = line_split[3]
                    if len(names) > 0:
                        names_split = names.split(',')
                        for name in names_split:
                            if len(name) > 0:
                                if contains_cyrillic(name):
                                    language_code = "ru"
                                    if language_code not in self.data[city_id]:
                                        self.data[city_id][language_code] = []
                                    if name not in self.data[city_id][language_code]:
                                        self.data[city_id][language_code].append(name)

    def save(self, filename: str = 'out\cities_db.json', add_timestamp: bool = True):
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            filename = filename.replace('.json', f'_{timestamp}.json')

        with open(os.path.join(self.path, filename), 'w', encoding='utf-8') as json_file:
            json_text = json.dumps(self.data, ensure_ascii=False, indent=4)
            json_file.write(json_text)

    def load(self, filename: str = 'out\cities_db.json'):
        with open(os.path.join(self.path, filename), 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def get_cities(self, language: str = 'en'):
        cities = []
        for city_id, city_names in self.data.items():
            if language in city_names:
                cities.extend(city_names[language])
        return cities

    def translate_city_name(self, city_name: str, target_language_code: str) -> Optional[List[str]]:
        for current_city_id, current_city_entry in self.data.items():
            for current_language_code, current_language_city_names in current_city_entry.items():
                if city_name in current_language_city_names and target_language_code in current_city_entry:
                    return current_city_entry[target_language_code][0]
        return None
