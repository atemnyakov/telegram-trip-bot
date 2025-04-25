import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset


class AnyDestinationClassifier:
    def __init__(self, path: str = f'./models/AnyDestinationClassifier'):
        self.path: str = path
        self.model: BertForSequenceClassification | None = None
        self.tokenizer: BertTokenizer | None = None

    def learn(self) -> None:
        # 1️⃣ Загружаем предобученный токенизатор
        self.tokenizer = BertTokenizer.from_pretrained("cointegrated/rubert-tiny")

        # 2️⃣ Данные в виде словаря {текст: лейбл}
        train_data = {
            "Хочу полететь куда угодно": 1,
            "Где дешёвые билеты в любую точку?": 1,
            "Любая страна, мне всё равно": 1,
            "Куда-то, но не знаю куда": 1,
            "В любую сторону, лишь бы улететь": 1,
            "Хочу полететь из Праги куда-то на выходных": 1,
            "Хочу на выходных улететь куда-то": 1,
            "Предложи мне путешествия из Праги на выходных до 2000 крон": 1,

            "Хочу в Париж": 0,
            "Ищу билеты в Лондон": 0,
            "Где дешёвые билеты в Токио?": 0,
            "Мне нужен билет в Москву": 0,
            "Как улететь в Нью-Йорк?": 0,
            "Найди билеты мне из Лондона до Рима": 0
        }

        # 3️⃣ Преобразуем словарь в списки
        train_texts = list(train_data.keys())  # Тексты (список)
        train_labels = list(train_data.values())  # Метки (список)

        # 4️⃣ Токенизируем
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=32, return_tensors="pt")

        # 5️⃣ Кастомный Dataset
        class FlightDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        train_dataset = FlightDataset(train_encodings, train_labels)

        # 6️⃣ Загружаем BERT
        self.model = BertForSequenceClassification.from_pretrained("cointegrated/rubert-tiny", num_labels=2)

        # 7️⃣ Настройки обучения
        training_args = TrainingArguments(
            output_dir="../results",
            num_train_epochs=10,
            per_device_train_batch_size=2,
            logging_dir="./logs",
            logging_steps=1,
            save_strategy="no"
        )

        # 8️⃣ Обучаем модель
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

    def save(self) -> None:
        # Save model and tokenizer
        self.model.save_pretrained(self.path)
        self.tokenizer.save_pretrained(self.path)

    def load(self) -> None:
        # Load the trained model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(self.path)
        self.tokenizer = BertTokenizer.from_pretrained(self.path)

    def test(self) -> None:
        # 9️⃣ Тестируем
        test_data = {
            "Хочу в Барселону": 0,
            "Есть ли какие-то билеты в Милан из Праги за 50 евро на выходных?": 0,
            "Куда-нибудь, неважно куда": 1,
            "Я хочу на выходных полететь куда-то": 1,
            "Я хочу на выходных полететь куда-то по Европе": 1,
            "Куда я могу полететь из Праги на выходных за 100 евро?": 1,
            "Куда я можно улететь из Вены за тысячу форинтов?": 1,
        }

        test_texts = list(test_data.keys())
        test_labels = list(test_data.values())

        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=32, return_tensors="pt")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**test_encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)

        #  🔥 Выводим предсказания
        print("Expected:", test_labels)
        print("Predicted:", predictions.tolist())

    def predict(self, text: str) -> bool:
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=32, return_tensors="pt")

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        # Print result
        print("Prediction:", prediction)  # 1 (any destination) or 0 (specific destination)
        return prediction
