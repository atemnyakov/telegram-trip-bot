from NNParserBase.NNParserBase import NNParserBase
from typing import Optional, List
from transformers import pipeline
import os


class PriceParser(NNParserBase):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def replace_currency_words_with_codes(self, query: str):
        words2codes = {
            "евро": "EUR",
            "крон": "CZK",
            "крона": "CZK"
        }
        for word, code in words2codes.items():
            query = query.replace(word, code)
        return query

    def create_tokens(self):
        tokens = []
        currencies = {"EUR", "CZK"}
        for i in range(1, 5000):
            for currency in currencies:
                token = f"{i} {currency}"
                tokens.append(token)
        return tokens

    def get_label_list(self) -> List[str]:
        return ["O", "EUR", "CZK"]

    def predict(self, text: str) -> Optional[str]:
        self.model.eval()
        ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        def parse_price(text):
            text = self.replace_currency_words_with_codes(text)
            results = ner_pipeline(text)
            price = [r["word"] for r in results if r["entity"] in ["CZK", "EUR"]]
            return price[0] if len(price) > 0 else None

        return parse_price(text)
