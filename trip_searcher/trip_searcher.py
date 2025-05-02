from city_db.city_db import CityDB
from date_classifier.date_classifier import DateClassifier
from price_parser.price_parser import PriceParser
from route_parser.route_parser import RouteParser
from flight_db.flight_db import FlightDB, FlightSearchParametersBase, SingleOriginDestinationSearchParameters, MultiOriginDestinationSearchParameters, Price
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

    def search(self, query: str) -> List[Tuple[Flight, Flight]]:
        parsed_route = self.route_parser.predict(query)
        parsed_dates = self.date_classifier.predict(query)
        parsed_price = self.price_parser.predict(query)

        # Parse cities
        origin_cities_ru = [city for city in parsed_route.get('B-DEP', [])]
        destination_cities_ru = [city for city in parsed_route.get('B-DEST', [])]

        # Initialize translation dictionaries
        cities_ru_to_en = {}
        cities_en_to_ru = {}
        codes_to_cities_en = {}
        cities_en_to_codes = {}

        # Collect all unique cities
        cities_ru = set(origin_cities_ru + destination_cities_ru)

        # Translate cities
        for city_ru in cities_ru:
            city_en = self.city_db.translate_city_name(city_name=city_ru, target_language_code='en')
            if city_en:
                cities_ru_to_en[city_ru] = city_en
                cities_en_to_ru[city_en] = city_ru

        # Get dates
        departure_date_from = parsed_dates.get("B-STARTDATE", datetime.now())
        departure_date_to = parsed_dates.get("B-ENDDATE", departure_date_from + timedelta(days=15))
        return_date_from = departure_date_from
        return_date_to = departure_date_to

        # Parse price
        price = None
        if parsed_price:
            price_value, price_currency = parsed_price.split()
            price = Price(currency=price_currency, value=float(price_value))

        # Initialize flight search parameters
        flight_search_params_base = FlightSearchParametersBase()
        flight_search_params_base.outbound_departure_date_from = departure_date_from
        flight_search_params_base.outbound_departure_date_to = departure_date_to
        flight_search_params_base.inbound_departure_date_from = return_date_from
        flight_search_params_base.inbound_departure_date_to = return_date_to
        flight_search_params_base.max_price = price
        flight_search_params_base.max_trip_duration = 3

        # Create single origin-destination search parameters
        single_origin_dest_search_params = SingleOriginDestinationSearchParameters()
        for attr, value in flight_search_params_base.__dict__.items():
            setattr(single_origin_dest_search_params, attr, value)

        # Search for flights
        for origin_city_ru in origin_cities_ru:
            origin_city_en = cities_ru_to_en[origin_city_ru]
            origin_code = self.flight_db.get_airport_code(origin_city_en)
            codes_to_cities_en[origin_code] = origin_city_en
            cities_en_to_codes[origin_city_en] = origin_code

            # Search for destinations
            if destination_cities_ru:
                for destination_city_ru in destination_cities_ru:
                    destination_city_en = cities_ru_to_en[destination_city_ru]
                    destination_code = self.flight_db.get_airport_code(destination_city_en)
                    codes_to_cities_en[destination_code] = destination_city_en
                    cities_en_to_codes[destination_city_en] = destination_code

                    # Set origin and destination for the search
                    single_origin_dest_search_params.origin = origin_code
                    single_origin_dest_search_params.destination = destination_code
                    self.flight_db.fetch_flights(single_origin_dest_search_params)
            else:
                # Only origin provided, no destination
                single_origin_dest_search_params.origin = origin_code
                single_origin_dest_search_params.destination = None
                self.flight_db.fetch_flights(single_origin_dest_search_params)

        # Create multi-origin-destination search parameters
        multi_origin_dest_search_params = MultiOriginDestinationSearchParameters()
        for attr, value in flight_search_params_base.__dict__.items():
            setattr(multi_origin_dest_search_params, attr, value)

        # Add origins to multi-origin search
        for origin_city_ru in origin_cities_ru:
            origin_city_en = cities_ru_to_en[origin_city_ru]
            origin_code = cities_en_to_codes.get(origin_city_en)
            multi_origin_dest_search_params.origins.append(origin_code)

        # Add destinations to multi-origin search if present
        if destination_cities_ru:
            for destination_city_ru in destination_cities_ru:
                destination_city_en = cities_ru_to_en[destination_city_ru]
                destination_code = cities_en_to_codes.get(destination_city_en)
                multi_origin_dest_search_params.destinations.append(destination_code)

        # Fetch flights for multi-origin-destination search
        round_trips = list(self.flight_db.get_flights(multi_origin_dest_search_params))
        round_trips.sort(key=lambda flight_pair: sum(flight.price.value for flight in flight_pair))

        return round_trips


if __name__ == '__main__':
    trip_searcher = TripSearcher(train_route_parser=True, train_date_classifier=True, train_price_parser=True)
    # trip_searcher.search(query='Куда я могу полететь из Праги или Вены на выходных?')
    trip_searcher.search(query='Куда я могу полететь из Праги или Вены с 1 мая по 1 июня с бюджетом до 100 евро?')


