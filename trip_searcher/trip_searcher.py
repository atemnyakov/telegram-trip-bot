import random
from city_db.city_db import CityDB
from date_classifier.date_classifier import DateClassifier
from price_parser.price_parser import PriceParser
from route_parser.route_parser import RouteParser
from trip_searcher.flight_db import FlightDB, FlightSearchParameters, Price
from datetime import datetime, timedelta


class TripSearcher:
    def __init__(self, train_route_parser: bool = False, train_date_classifier: bool = False, train_price_parser: bool = False):
        self.route_parser = RouteParser()
        if train_route_parser:
            self.route_parser.learn()
            self.route_parser.save()
        else:
            self.route_parser.load()

        self.date_classifier = DateClassifier()
        if train_date_classifier:
            self.date_classifier.learn()
            self.date_classifier.save()
        else:
            self.date_classifier.load()

        self.price_parser = PriceParser()
        if train_price_parser:
            self.price_parser.learn()
            self.price_parser.save()
        else:
            self.price_parser.load()

        self.city_db = CityDB()
        self.city_db.load()

        self.flight_db = FlightDB()
        self.flight_db.load_airport_codes()
        # self.flight_db.load_flights()

    def search(self, query: str):
        parsed_route = self.route_parser.predict(query)
        parsed_dates = self.date_classifier.predict(query)
        parsed_price = self.price_parser.predict(query)

        origin_cities_ru = [origin_city for origin_city in parsed_route['B-DEP'] if 'B-DEP' in parsed_route]

        destination_cities_ru = [destination_city for destination_city in parsed_route['B-DEST'] if 'B-DEST' in parsed_route]

        cities_ru_to_en = dict()

        cities_ru = set()
        cities_ru.update(origin_cities_ru)
        cities_ru.update(destination_cities_ru)

        for city_ru in cities_ru:
            city_en = self.city_db.translate_city_name(city_name=city_ru, target_language_code='en')
            if city_en:
                cities_ru_to_en[city_ru] = city_en

        update_flight_database = True

        # Get the current date
        departure_date_from = datetime.now()
        if "B-STARTDATE" in parsed_dates:
            departure_date_from = parsed_dates["B-STARTDATE"]

        departure_date_to = departure_date_from + timedelta(days=15)
        if "B-ENDDATE" in parsed_dates:
            departure_date_to = parsed_dates["B-ENDDATE"]

        return_date_from = departure_date_from

        return_date_to = departure_date_to

        price = None
        if parsed_price is not None:
            price_value, price_currency = parsed_price.split()
            price = Price(currency=price_currency, value=float(price_value))

        round_trips = []

        for origin_city_ru in origin_cities_ru:
            origin_city_en = cities_ru_to_en[origin_city_ru]
            origin_code = self.flight_db.get_airport_code(origin_city_en)
            if len(destination_cities_ru) > 0:
                for destination_city_ru in destination_cities_ru:
                    destination_city_en = cities_ru_to_en[destination_city_ru]
                    destination_code = self.flight_db.get_airport_code(destination_city_en)

                    flight_search_parameters = FlightSearchParameters()
                    flight_search_parameters.origin = origin_code
                    flight_search_parameters.destination = destination_code
                    flight_search_parameters.outbound_departure_date_from = departure_date_from
                    flight_search_parameters.outbound_departure_date_to = departure_date_to
                    flight_search_parameters.inbound_departure_date_from = return_date_from
                    flight_search_parameters.inbound_departure_date_to = return_date_to
                    flight_search_parameters.max_price = price
                    flight_search_parameters.max_trip_duration = 3

                    self.flight_db.fetch_flights(flight_search_parameters)
                    round_trips.extend(self.flight_db.get_flights(flight_search_parameters))
            else:
                flight_search_parameters = FlightSearchParameters()
                flight_search_parameters.origin = origin_code
                flight_search_parameters.outbound_departure_date_from = departure_date_from
                flight_search_parameters.outbound_departure_date_to = departure_date_to
                flight_search_parameters.inbound_departure_date_from = return_date_from
                flight_search_parameters.inbound_departure_date_to = return_date_to
                flight_search_parameters.max_price = price
                flight_search_parameters.max_trip_duration = 3

                self.flight_db.fetch_flights(flight_search_parameters)
                round_trips.extend(self.flight_db.get_flights(flight_search_parameters))

        response = ""

        round_trips.sort(key=lambda round_flight: round_flight[0].price.value + round_flight[1].price.value)

        for i in range(min(10, len(round_trips))):
            outbound_flight = round_trips[i][0]
            inbound_flight = round_trips[i][1]

            response += f"Trip | Outbound flight: {str(outbound_flight)} | Inbound flight: {str(inbound_flight)}\n\n"

        return response


if __name__ == '__main__':
    trip_searcher = TripSearcher(train_route_parser=False, train_date_classifier=False, train_price_parser=False)
    # trip_searcher.search(query='Куда я могу полететь из Праги или Вены на выходных?')
    trip_searcher.search(query='Куда я могу полететь из Праги или Вены с 1 мая по 1 июня с бюджетом до 100 евро?')


