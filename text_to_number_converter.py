import pymorphy3
import re

class TextToNumberConverter:
    def __init__(self):
        self.texts_to_number = {
            "один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
            "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
            "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
            "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
            "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
            "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
            "тысяча": 1000, "миллион": 1000000
        }
        self.morph = pymorphy3.MorphAnalyzer()

    def convert(self, text: str):
        tokens = text.split()
        tokens = [[parsed_token.word, parsed_token.normal_form] for token in tokens if (parsed_token := self.morph.parse(token)[0])]
        number_of_tokens = len(tokens)
        i = 0
        while i < number_of_tokens:
            token = tokens[i]
            if token[1] in self.texts_to_number:
                partial_number_tokens = [token]
                j = i + 1
                while j < number_of_tokens:
                    next_token = tokens[j]
                    if next_token[1] in self.texts_to_number:
                        partial_number_tokens.append(next_token)
                        j += 1
                    else:
                        break
                result_number = 0
                for current_token in partial_number_tokens:
                    current_token_number = self.texts_to_number[current_token[1]]
                    if current_token_number >= 1000:
                        result_number *= current_token_number
                    else:
                        result_number += current_token_number
                i += (j - i)
                result_number_text = " ".join([token[0] for token in partial_number_tokens])
                text = re.sub(result_number_text, f"{result_number}", text, flags=re.IGNORECASE)
            else:
                i += 1



