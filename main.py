from any_destination_classifier import AnyDestinationClassifier
from route_parser.route_parser import RouteParser
from ryanair import Ryanair
from date_classifier.date_classifier import DateClassifier
from wizzair import Wizzair


def test_ryanair():
    ryanair = Ryanair()

    airport_info = ryanair.get_airport_info('PRG')
    print(airport_info)

    destinations = ryanair.get_destinations(origin='PRG')
    print(destinations)

    cheapest_flights = ryanair.get_cheapest_flights(
        origin='PRG',
        destination='BCN',
        departure_date='2025-03-01'
    )
    print(cheapest_flights)

    one_way_fares = ryanair.get_one_way_fares(
        origin='PRG',
        outbound_departure_date_from='2025-06-01',
        outbound_departure_date_to='2025-06-29'
    )
    print(one_way_fares)

    round_trip_fares = ryanair.get_round_trip_fares(
        origin='PRG',
        outbound_departure_date_from='2025-06-01',
        outbound_departure_date_to='2025-06-29',
        inbound_departure_date_from='2025-07-03',
        inbound_departure_date_to='2025-07-31'
    )
    print(round_trip_fares)

    one_way_flights = ryanair.get_flights(
        origin='PRG',
        destination='BCN',
        departure_date='2025-03-03'
    )
    print(one_way_flights)

    round_flights = ryanair.get_flights(
        origin='PRG',
        destination='BCN',
        departure_date='2025-03-03',
        return_date='2025-03-06'
    )
    print(round_flights)


def test_wizzair():
    wizzair = Wizzair()

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO'
    )
    print(flights)

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO',
        departure_date='2025-03-14'
    )
    print(flights)

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO',
        months=['2025-03']
    )
    print(flights)

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO',
        departure_date='2025-03-14',
        return_date='2025-03-16'
    )
    print(flights)

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO',
        departure_date='2025-03-14',
        trip_duration='1-3 days'
    )
    print(flights)

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO',
        departure_date='2025-03-14',
        trip_duration='4-8 days'
    )
    print(flights)

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO',
        months=['2025-03'],
        trip_duration='1-3 days'
    )
    print(flights)

    flights = wizzair.get_flights(
        origin='PRG',
        destination='FCO',
        months=['2025-03'],
        trip_duration='anytime'
    )
    print(flights)

    flights = wizzair.get_timetable(
        origin='PRG',
        destination='FCO',
        outbound_departure_date_from='2025-02-26',
        outbound_departure_date_to='2025-04-06',
        inbound_departure_date_from='2025-02-26',
        inbound_departure_date_to='2025-04-06'
    )
    print(flights)

    farechart = wizzair.get_farechart(
        origin='PRG',
        destination='FCO',
        departure_date='2025-03-14',
        return_date='2025-03-16'
    )
    print(farechart)


class Coordinates:
    """
    Represents geographical coordinates with latitude and longitude.
    """

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def __str__(self):
        return f"({self.latitude}, {self.longitude})"

    def __repr__(self):
        return f"Coordinates(latitude={self.latitude}, longitude={self.longitude})"


class Country:
    """
    Represents a country with a unique code, name, and currency.
    """

    def __init__(self, code, name):
        self.code = code
        self.name = name

    def __str__(self):
        return f"{self.name} ({self.code})"

    def __repr__(self):
        return f"Country(code={self.code!r}, name={self.name!r})"


class Price:
    """
    Represents a price with a currency and value.
    """

    def __init__(self, currency, value):
        self.currency = currency
        self.value = value

    def __str__(self):
        return f"{self.value} {self.currency}"

    def __repr__(self):
        return f"Price(currency={self.currency!r}, value={self.value})"

    def __eq__(self, other):
        if not isinstance(other, Price):
            return False
        return (
                self.value == other.value and
                self.currency == other.currency
        )

    def __hash__(self):
        return hash((
            self.value, self.currency
        ))


class Flight:
    """
    Represents a flight with an origin, destination, departure date, and price.
    """

    def __init__(self, origin, destination, departure_date, arrival_date, airline, price):
        self.origin = origin
        self.destination = destination
        self.departure_date = departure_date
        self.arrival_date = arrival_date
        self.airline = airline
        self.price = price

    def __str__(self):
        return f"From {self.origin} to {self.destination}, " \
               f"departure: {self.departure_date}, arrival: {self.arrival_date}, " \
               f"airline: {self.airline}, price: {self.price}"

    def __repr__(self):
        return (f"Flight(origin={self.origin!r}, destination={self.destination!r}, "
                f"departure_date={self.departure_date!r}, arrival_date={self.arrival_date}, "
                f"airline={self.airline}, price={self.price!r})")

    def __eq__(self, other):
        if not isinstance(other, Flight):
            return False
        return (
                self.origin == other.origin and
                self.destination == other.destination and
                self.departure_date == other.departure_date and
                self.arrival_date == other.arrival_date and
                self.airline == other.airline and
                self.price == other.price
        )

    def __hash__(self):
        return hash((
            self.origin, self.destination,
            self.departure_date, self.arrival_date,
            self.airline, self.price
        ))


def test():
    origins = ['PRG', 'BRQ', 'VIE']

    outbound_departure_date_from = '2025-03-15'
    outbound_departure_date_to = '2025-03-31'
    inbound_departure_date_from = '2025-03-15'
    inbound_departure_date_to = '2025-03-31'

    outbound_flights = []
    inbound_flights = []
    round_flights = []

    ryanair = Ryanair()

    for origin in origins:
        def create_flight(flight):
            return Flight(
                origin=flight.origin,
                destination=flight.destination,
                departure_date=flight.departure_date,
                arrival_date=flight.arrival_date,
                airline='Ryanair',
                price=Price(flight.price.currency, flight.price.value)
            )

        current_origin_round_trips = ryanair.get_round_trip_fares(
            origin=origin,
            outbound_departure_date_from=outbound_departure_date_from,
            outbound_departure_date_to=outbound_departure_date_to,
            inbound_departure_date_from=inbound_departure_date_from,
            inbound_departure_date_to=inbound_departure_date_to,
            currency='CZK',
            adult_pax_count=1,
            market='cs-cz',
            duration_from=0,
            duration_to=3
        )

        outbound_flights += [create_flight(outbound) for outbound, _ in current_origin_round_trips]
        inbound_flights += [create_flight(inbound) for _, inbound in current_origin_round_trips]

    wizzair = Wizzair()

    for origin in origins:
        route_map = wizzair.get_map()

        if origin in route_map[1]:
            destinations = route_map[1][origin]

            for destination in destinations:
                def create_flight(flight):
                    return Flight(
                        origin=flight.origin,
                        destination=flight.destination,
                        departure_date=flight.departure_date,
                        arrival_date=None,
                        airline='Wizzair',
                        price=Price(flight.price.currency, flight.price.value)
                    )

                timetable = wizzair.get_timetable(
                    origin=origin,
                    destination=destination,
                    outbound_departure_date_from=outbound_departure_date_from,
                    outbound_departure_date_to=outbound_departure_date_to,
                    inbound_departure_date_from=inbound_departure_date_from,
                    inbound_departure_date_to=inbound_departure_date_to,
                )

                outbound_flights += [create_flight(flight) for flight in timetable[0]]
                inbound_flights += [create_flight(flight) for flight in timetable[1]]

    def has_duplicates(lst):
        return len(lst) != len(set(lst))

    print(has_duplicates(outbound_flights))
    print(has_duplicates(inbound_flights))

    from collections import Counter

    def remove_duplicates(lst):
        counts = Counter(lst)
        return [item for item in lst if counts[item] == 1]

    outbound_flights = remove_duplicates(outbound_flights)
    inbound_flights = remove_duplicates(inbound_flights)

    print(has_duplicates(outbound_flights))
    print(has_duplicates(inbound_flights))

    def convert_currency(amount, from_currency, to_currency):
        from currency_converter import CurrencyConverter
        c = CurrencyConverter()
        converted_amount = c.convert(amount, from_currency, to_currency)
        return converted_amount

    for outbound in outbound_flights:
        for inbound in inbound_flights:
            # Check if the outbound flight and inbound flight are a valid pair (same origin and destination)
            if outbound.destination == inbound.origin:
                # Convert string departure date to datetime object
                from datetime import datetime
                outbound_departure = datetime.strptime(outbound.departure_date, "%Y-%m-%dT%H:%M:%S")
                inbound_departure = datetime.strptime(inbound.departure_date, "%Y-%m-%dT%H:%M:%S")

                # Check conditions for the outbound flight's start day and trip duration
                departure_day = outbound_departure.weekday()  # 0: Monday, 6: Sunday

                trip_duration = (inbound_departure - outbound_departure).days

                # Condition for start date and trip duration
                if ((departure_day == 3) and (1 <= trip_duration <= 4)) or \
                        ((departure_day == 4) and (1 <= trip_duration <= 3)) or \
                        ((departure_day == 5) and (1 <= trip_duration <= 2)) or \
                        ((departure_day == 6) and (trip_duration == 1)):
                    # Check if the round flight's price is less than or equal to the max price
                    outbound_price = outbound.price.value if outbound.price.currency == 'CZK' else convert_currency(outbound.price.value, outbound.price.currency, 'CZK')
                    inbound_price = inbound.price.value if inbound.price.currency == 'CZK' else convert_currency(inbound.price.value, inbound.price.currency, 'CZK')
                    total_price = outbound_price + inbound_price
                    if total_price <= 5000:
                        round_flights.append((outbound, inbound))


class TripParameters:
    def __init__(self):
        self.origin: str | None = None
        self.destination: str | None = None
        self.budget: Price | None = None


def test_ai():
    import re
    import spacy
    import dateparser
    from transformers import pipeline

    # Загружаем spaCy для NER
    nlp = spacy.load("ru_core_news_sm")

    # Hugging Face NER для улучшенного извлечения сущностей
    ner_pipeline = pipeline("ner", model="Gherman/bert-base-NER-Russian")

    def extract_dates(text):
        """Парсинг дат и относительных временных выражений."""
        if "выходные" in text:
            return [5, 6, 7]  # Пт-Сб-Вс

        date = dateparser.parse(text, languages=["ru"])
        if date:
            return [date.weekday() + 1]  # Приводим к [1, 2, ... 7]

        return []

    def parse_flight_query(query):
        """Основной обработчик текста запроса."""
        data = {
            "origin": None,
            "destination": "any",
            "departure_days": [],
            "return_days": [],
            "trip_duration": [],
            "budget": None
        }

        # Парсим бюджет
        budget_match = re.search(r'(\d+)\s*(?:крон|руб|€|\$)', query)
        if budget_match:
            data["budget"] = int(budget_match.group(1))

        # NLP-анализ
        doc = nlp(query)

        # Извлекаем города
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:  # GPE = города и страны
                if "из" in query[:ent.start_char]:  # Определяем отправную точку
                    data["origin"] = ent.text
                else:
                    data["destination"] = ent.text

        # HuggingFace NER (доп. проверка)
        ner_results = ner_pipeline(query)
        for entity in ner_results:
            if entity["entity"] == "B-LOC" and data["origin"] is None:
                data["origin"] = entity["word"]

        # Определяем дни вылета
        data["departure_days"] = extract_dates(query)
        data["return_days"] = data["departure_days"]  # По умолчанию, если не указано другое

        return data

    # Тест
    query = "Я хочу куда-то полететь из Праги на выходных, имея бюджет 2000 крон."
    # query = "Меня зовут Сергей Иванович из Вены."
    # doc = nlp(query)
    # lemmas = [token.lemma_ for token in doc]
    # query = " ".join(lemmas)
    def lemmatize_text_spacy(text):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])
    query = lemmatize_text_spacy(query)
    print(parse_flight_query(query))


def test_pymorhpy():
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()

    # Список городов в именительном падеже
    city_list = {"Санкт-Петербург", "Новая Каховка", "Москва", "Екатеринбург", "Казань"}

    morph = pymorphy3.MorphAnalyzer()

    def normalize_city_name(city_name):
        words = city_name.split()

        # Определяем род первого слова (важно для прилагательных типа "Новая")
        parsed_first_word = morph.parse(words[0])[0]
        gender = parsed_first_word.tag.gender  # 'masc', 'femn', 'neut' или None

        normalized_words = []
        for word in words:
            parsed = morph.parse(word)[0]

            if gender:
                # Приводим слово в именительный падеж с учетом рода
                normal_form = parsed.inflect({'nomn', gender})
            else:
                normal_form = parsed.inflect({'nomn'})

            if normal_form:
                normalized_words.append(normal_form.word)
            else:
                normalized_words.append(parsed.normal_form)

        normalized_city = " ".join(normalized_words).title()

        # Проверяем, есть ли нормализованный вариант в списке городов
        if normalized_city in city_list:
            return normalized_city
        return city_name  # Оставляем как есть, если не нашли

    def normalize_text_cities(text):
        words = text.split()
        normalized_words = []

        i = 0
        while i < len(words):
            found = False
            for length in range(3, 0, -1):  # Проверяем фразы длиной от 3 до 1 слова
                phrase = " ".join(words[i:i + length])
                normalized_phrase = normalize_city_name(phrase)
                if normalized_phrase in city_list:
                    normalized_words.append(normalized_phrase)
                    i += length  # Пропускаем использованные слова
                    found = True
                    break
            if not found:
                normalized_words.append(words[i])
                i += 1

        return " ".join(normalized_words)

    # Тестовые примеры
    text1 = "Я хочу посетить музей в Санкт-Петербурге"
    text2 = "Я из Новой Каховки"
    text3 = "Я хочу посетить музей Франкфурта-на-Майне"

    print(normalize_text_cities(text1))  # "Я хочу посетить музей в Санкт-Петербург"
    print(normalize_text_cities(text2))  # "Я из Новая Каховка"
    print(normalize_text_cities(text3))


def test_any_dest_classifier():
    any_destination_classifier = AnyDestinationClassifier()
    # any_destination_classifier.learn()
    # any_destination_classifier.save()
    # any_destination_classifier.test()
    any_destination_classifier.load()
    any_destination_classifier.test()
    # any_destination_classifier.predict(text="Я хочу улететь куда-нибудь")


def test_date_classifier():
    date_classifier = DateClassifier()
    # date_classifier.learn()
    # date_classifier.save()
    date_classifier.load()
    date_classifier.test()
    queries = [
        "Я хочу куда-то полететь с двадцать первого января по 22 января",
        "Я хочу куда-то полететь по двадцать второго января с 21 января",
        "В Берлине буду с 1 марта до 3 марта",
        "Буду в Берлине до 4 апреля"
    ]
    for query in queries:
        start_end_date = date_classifier.predict(query)
        print(start_end_date)
    # date_classifier.replace_texts_with_numbers(text="Я хочу полететь двадцать первого января в отпуск")


def test_queries():
    dep_arr_classifier = RouteParser()
    date_classifier = DateClassifier()


if __name__ == '__main__':
    # test_queries()

    # text_to_number_converter = TextToNumberConverter()
    # text_to_number_converter.convert("Две тысячи пятьсот один и три тысячи два и сто двадцать пять тысяч триста пятьдесят три")



