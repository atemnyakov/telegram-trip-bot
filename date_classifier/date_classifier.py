from NNParserBase.neural_parser_base import NeuralParserBase
from datetime import datetime
from typing import Dict, Optional, List
from transformers import pipeline
import calendar
import os


class DateClassifier(NeuralParserBase):
    def __init__(self):
        super().__init__(path=os.path.dirname(__file__))

    def create_tokens(self):
        months_in_russian = [
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ]
        dates = []
        for month in range(1, 13):
            days_in_month = calendar.monthrange(2025, month)[1]
            for day in range(1, days_in_month + 1):
                dates.append(f"{day} {months_in_russian[month - 1]}")
        return dates

    def get_label_list(self) -> List[str]:
        return ["O", "B-STARTDATE", "B-ENDDATE"]

    def predict(self, query: str) -> Dict[str, Optional[datetime]]:
        self.model.eval()
        ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        def parse_dates(query: str):
            results = ner_pipeline(query)

            def datetime_from_text(text: str, end_of_day: bool) -> datetime:
                tokens = text.split()
                months_to_numbers = {
                    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5,
                    "июня": 6, "июля": 7, "августа": 8, "сентября": 9, "октября": 10,
                    "ноября": 11, "декабря": 12,
                }

                month_and_day_str = f"{months_to_numbers[tokens[1]]}-{tokens[0]}"
                year = datetime.now().year
                date = datetime.strptime(f"{year}-{month_and_day_str}", "%Y-%m-%d")
                if end_of_day:
                    date = date.replace(hour=23, minute=59)
                return date

            start_dates = [datetime_from_text(r["word"], False) for r in results if r["entity"] == "B-STARTDATE"]
            end_dates = [datetime_from_text(r["word"], True) for r in results if r["entity"] == "B-ENDDATE"]

            return {"B-STARTDATE": start_dates[0] if len(start_dates) > 0 else None, "B-ENDDATE": end_dates[0] if len(end_dates) > 0 else None}

        return parse_dates(query)