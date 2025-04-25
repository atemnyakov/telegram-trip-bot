from city_db.city_db import CityDB
from route_parser.route_parser import RouteParser
from flight_db import FlightDB, SearchFlightParameters
from datetime import datetime, timedelta


class TripSearcher:
    def __init__(self, train_route_parser: bool = False):
        self.route_parser = RouteParser()
        if train_route_parser:
            self.route_parser.learn()
            self.route_parser.save()
        else:
            self.route_parser.load()

        self.city_db = CityDB()
        self.city_db.load()

        self.flight_db = FlightDB()
        self.flight_db.load_airport_codes()
        self.flight_db.load_flights()

    def search(self, query: str):
        route_prediction = self.route_parser.predict(query)

        origin_cities_ru = [origin_city for origin_city in route_prediction['B-DEP'] if 'B-DEP' in route_prediction]

        destination_cities_ru = [destination_city for destination_city in route_prediction['B-DEST'] if 'B-DEST' in route_prediction]

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
        departure_date_from_str = departure_date_from.strftime("%y-%m-%d")

        departure_date_to = departure_date_from + timedelta(days=10)
        departure_date_to_str = departure_date_to.strftime("%y-%m-%d")

        return_date_from = departure_date_from + timedelta(days=5)
        return_date_from_str = return_date_from.strftime("%y-%m-%d")

        return_date_to = departure_date_from + timedelta(days=15)
        return_date_to_str = return_date_to.strftime("%y-%m-%d")

        round_trips = []

        for origin_city_ru in origin_cities_ru:
            origin_city_en = cities_ru_to_en[origin_city_ru]
            origin_code = self.flight_db.get_airport_code(origin_city_en)
            if len(destination_cities_ru) > 0:
                for destination_city_ru in destination_cities_ru:
                    destination_city_en = cities_ru_to_en[destination_city_ru]
                    destination_code = self.flight_db.get_airport_code(destination_city_en)

                    search_flight_parameters = SearchFlightParameters()
                    search_flight_parameters.origin = origin_code
                    search_flight_parameters.destination = destination_code
                    search_flight_parameters.outbound_departure_date_from = departure_date_from
                    search_flight_parameters.outbound_departure_date_to = departure_date_to
                    search_flight_parameters.inbound_departure_date_from = departure_date_from
                    search_flight_parameters.inbound_departure_date_to = departure_date_to

                    self.flight_db.fetch_flights(search_flight_parameters)
                    round_trips = self.flight_db.get_flights(search_flight_parameters)
            else:
                search_flight_parameters = SearchFlightParameters()
                search_flight_parameters.origin = origin_code
                search_flight_parameters.outbound_departure_date_from = departure_date_from
                search_flight_parameters.outbound_departure_date_to = departure_date_to
                search_flight_parameters.inbound_departure_date_from = departure_date_from
                search_flight_parameters.inbound_departure_date_to = departure_date_to

                self.flight_db.fetch_flights(search_flight_parameters)
                round_trips = self.flight_db.get_flights(search_flight_parameters)

        response = ""

        for round_trip in round_trips:
            outbound_flight = round_trip[0]
            inbound_flight = round_trip[1]

            response += f"Trip | Outbound flight: {str(outbound_flight)} | Inbound flight: {str(inbound_flight)}\n\n"

        return response


if __name__ == '__main__':
    trip_searcher = TripSearcher(train_route_parser=False)
    trip_searcher.search(query='Куда я могу полететь из Праги или Вены на выходных?')


