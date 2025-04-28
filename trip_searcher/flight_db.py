import copy
import json
import os
from datetime import datetime, timedelta
from typing import Set, Tuple, Dict, Optional

from currency_converter import CurrencyConverter

from ryanair import Ryanair
from wizzair import Wizzair


class Coordinates:
    """
    Represents geographical coordinates with latitude and longitude.
    """

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def __str__(self):
        return f"({self.latitude}, {self.longitude})"

    def __repr__(self):
        return f"Coordinates(latitude={self.latitude}, longitude={self.longitude})"

    def to_dict(self):
        return {"latitude": self.latitude, "longitude": self.longitude}

    @classmethod
    def from_dict(cls, data):
        return cls(data["latitude"], data["longitude"])


class Country:
    """
    Represents a country with a unique code, name, and currency.
    """

    def __init__(self, code: str, name: str):
        self.code = code
        self.name = name

    def __str__(self):
        return f"{self.name} ({self.code})"

    def __repr__(self):
        return f"Country(code={self.code!r}, name={self.name!r})"

    def to_dict(self):
        return {"code": self.code, "name": self.name}

    @classmethod
    def from_dict(cls, data):
        return cls(data["code"], data["name"])


class Price:
    """
    Represents a price with a currency and value.
    """

    def __init__(self, currency: str, value: float):
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

    def to_dict(self):
        return {"currency": self.currency, "value": self.value}

    @classmethod
    def from_dict(cls, data):
        return cls(data["currency"], data["value"])


class FlightSearchParameters:
    def __init__(self):
        self.origin: str | None = None
        self.outbound_departure_date_from: datetime | None = None
        self.outbound_departure_date_to: datetime | None = None
        self.destination: str | None = None
        self.inbound_departure_date_from: datetime | None = None
        self.inbound_departure_date_to: datetime | None = None
        self.max_price: Price | None = None
        self.min_trip_duration: int = 2
        self.max_trip_duration: int = 5


class Flight:
    """
    Represents a flight with an origin, destination, departure date, and price.
    """

    def __init__(self, origin: str, destination: str, departure_date: datetime, airline: str, price: Price):
        self.origin = origin
        self.destination = destination
        self.departure_date = departure_date
        self.airline = airline
        self.price = price

    def __str__(self):
        return f"From {self.origin} to {self.destination}, " \
               f"departure: {self.departure_date}, " \
               f"airline: {self.airline}, price: {self.price}"

    def __repr__(self):
        return (f"Flight(origin={self.origin!r}, destination={self.destination!r}, "
                f"departure_date={self.departure_date!r}, "
                f"airline={self.airline}, price={self.price!r})")

    def __eq__(self, other):
        if not isinstance(other, Flight):
            return False
        return (
                self.origin == other.origin and
                self.destination == other.destination and
                self.departure_date == other.departure_date and
                self.airline == other.airline and
                self.price == other.price
        )

    def __hash__(self):
        return hash((
            self.origin, self.destination,
            self.airline, self.price
        ))

    def to_dict(self):
        return {
            "origin": self.origin,
            "destination": self.destination,
            "departure_date": self.departure_date.isoformat(),
            "airline": self.airline,
            "price": self.price.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            origin=data["origin"],
            destination=data["destination"],
            departure_date=datetime.fromisoformat(data["departure_date"]),
            airline=data["airline"],
            price=Price.from_dict(data["price"])
        )


class FlightDB:
    def __init__(self, path: str = os.path.join(os.path.dirname(__file__), "out")):
        self.path: str = path
        self.airport_codes: Dict[str, Set[str]] = dict()
        self.flights: Set[Flight] = set()
        self.ryanair = Ryanair()
        self.wizzair = Wizzair()

    def save_airport_codes(self) -> None:
        full_path = os.path.join(self.path, "airport_code_db.json")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            serializable_dict = {k: list(v) for k, v in self.airport_codes.items()}
            json.dump(serializable_dict, f, ensure_ascii=False, indent=4)

    def load_airport_codes(self) -> bool:
        full_path = os.path.join(self.path, "airport_code_db.json")

        if not os.path.exists(full_path):
            return False

        try:
            with open(full_path, "r", encoding="utf-8") as f:
               loaded_dict = json.load(f)
               self.airport_codes = {k: set(v) for k, v in loaded_dict.items()}

            return True
        except:
            return False

    def fetch_airport_codes(self):
        destinations_wzz, connections_wzz = self.wizzair.get_map()

        for destination_wzz in destinations_wzz:
           self.airport_codes.setdefault(destination_wzz.code, set()).add(destination_wzz.name)

        airports_ryr = self.ryanair.get_airports()
        for airport_ryr in airports_ryr:
            self.airport_codes.setdefault(airport_ryr.code, set()).add(airport_ryr.name)

    def get_airport_code(self, city_name: str) -> Optional[str]:
        for current_airport_code, current_city_names in self.airport_codes.items():
            if city_name in current_city_names:
                return current_airport_code
        return None

    def save_flights(self) -> None:
        full_path = os.path.join(self.path, "flight_db.json")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as json_file:
            json_data = [flight.to_dict() for flight in self.flights]
            json_text = json.dumps(json_data, ensure_ascii=False, indent=4)
            json_file.write(json_text)

    def load_flights(self) -> bool:
        full_path = os.path.join(self.path, "flight_db.json")

        if not os.path.exists(full_path):
            return False

        try:
            with open(full_path, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
                self.flights = set([Flight.from_dict(flight_data) for flight_data in json_data])

            return True
        except:
            return False

    def fetch_flights(self, parameters: FlightSearchParameters) -> None:
        outbound_flights = set()
        inbound_flights = set()

        # def create_flight_ryr(flight):
        #     return Flight(
        #         origin=flight.origin,
        #         destination=flight.destination,
        #         departure_date=flight.departure_date,
        #         arrival_date=flight.arrival_date,
        #         airline='Ryanair',
        #         price=Price(flight.price.currency, flight.price.value)
        #     )
        #
        # current_origin_round_trips = self.ryanair.get_flights(
        #     origin=origin,
        #     outbound_departure_date_from=outbound_departure_date_from,
        #     outbound_departure_date_to=outbound_departure_date_to,
        #     inbound_departure_date_from=inbound_departure_date_from,
        #     inbound_departure_date_to=inbound_departure_date_to,
        #     currency='CZK',
        #     adult_pax_count=1,
        #     market='cs-cz',
        #     duration_from=0,
        #     duration_to=3
        # )

        # outbound_flights += [create_flight_ryr(outbound) for outbound, _ in current_origin_round_trips]
        # inbound_flights += [create_flight_ryr(inbound) for _, inbound in current_origin_round_trips]

        def search_ryr(parameters_ryr: FlightSearchParameters):
            def create_flight(flight):
                return Flight(
                    origin=flight.origin,
                    destination=flight.destination,
                    departure_date=datetime.strptime(flight.departure_date, "%Y-%m-%dT%H:%M:%S"),
                    airline='Ryanair',
                    price=Price(flight.price.currency, flight.price.value)
                )

            nonlocal outbound_flights
            nonlocal inbound_flights

            cheapest_flights_outbound = self.ryanair.get_cheapest_flights(
                origin=parameters_ryr.origin,
                destination=parameters_ryr.destination,
                departure_date=parameters_ryr.outbound_departure_date_from.strftime("%Y-%m-%d")
            )

            outbound_flights.update([create_flight(flight) for flight in cheapest_flights_outbound])

            cheapest_flights_inbound = self.ryanair.get_cheapest_flights(
                origin=parameters_ryr.destination,
                destination=parameters_ryr.origin,
                departure_date=parameters_ryr.inbound_departure_date_from.strftime("%Y-%m-%d")
            )

            inbound_flights.update([create_flight(flight) for flight in cheapest_flights_inbound])

            # one_way_flights = self.ryanair.get_flights(
            #     origin=parameters_ryr.origin,
            #     destination=parameters_ryr.destination,
            #     departure_date=parameters_ryr.outbound_departure_date_from.strftime("%Y-%m-%d")
            # )
            # print(one_way_flights)

            flex_days_after_departure = (parameters_ryr.outbound_departure_date_to - parameters_ryr.outbound_departure_date_from).days
            flex_days_after_return = (parameters_ryr.inbound_departure_date_to - parameters_ryr.inbound_departure_date_from).days

            # step_days = 5
            # step = timedelta(days=step_days)
            #
            # current_outbound_departure_date_from = parameters_ryr.outbound_departure_date_from
            #
            # while current_outbound_departure_date_from < parameters_ryr.outbound_departure_date_to:
            #     current_outbound_departure_date_to = min(current_outbound_departure_date_from + step, parameters_ryr.outbound_departure_date_to)
            #
            #     current_inbound_departure_date_from = parameters_ryr.inbound_departure_date_from
            #
            #     while current_inbound_departure_date_from < parameters_ryr.inbound_departure_date_to:
            #         current_inbound_departure_date_to = min(current_inbound_departure_date_from + step, parameters_ryr.inbound_departure_date_to)
            #
            #         round_flights = self.ryanair.get_flights(
            #             origin=parameters_ryr.origin,
            #             destination=parameters_ryr.destination,
            #             departure_date=current_outbound_departure_date_from.strftime("%Y-%m-%d"),
            #             flex_days_before_departure=0,
            #             flex_days_after_departure=step_days,
            #             return_date=current_inbound_departure_date_from.strftime("%Y-%m-%d"),
            #             flex_days_before_return=0,
            #             flex_days_after_return=step_days
            #         )
            #
            #         if round_flights is not None and len(round_flights) > 0 and round_flights[0] is not None and len(
            #                 round_flights[0]) > 0:
            #             outbound_flights.update([create_flight(flight) for flight in round_flights[0]])
            #
            #         if round_flights is not None and len(round_flights) > 1 and round_flights[1] is not None and len(
            #                 round_flights[1]) > 0:
            #             inbound_flights.update([create_flight(flight) for flight in round_flights[1]])
            #
            #         current_inbound_departure_date_from = current_inbound_departure_date_to
            #
            #     current_outbound_departure_date_from = current_outbound_departure_date_to

        def search_wzz(parameters_wzz: FlightSearchParameters):
            timetable = self.wizzair.get_timetable(
                origin=parameters_wzz.origin,
                destination=parameters_wzz.destination,
                outbound_departure_date_from=parameters_wzz.outbound_departure_date_from.strftime("%Y-%m-%d"),
                outbound_departure_date_to=parameters_wzz.outbound_departure_date_to.strftime("%Y-%m-%d"),
                inbound_departure_date_from=parameters_wzz.inbound_departure_date_from.strftime("%Y-%m-%d"),
                inbound_departure_date_to=parameters_wzz.inbound_departure_date_to.strftime("%Y-%m-%d"),
            )

            def create_flight(flight):
                return Flight(
                    origin=flight.origin,
                    destination=flight.destination,
                    departure_date=datetime.strptime(flight.departure_date, "%Y-%m-%dT%H:%M:%S"),
                    airline='Wizzair',
                    price=Price(flight.price.currency, flight.price.value)
                )

            nonlocal outbound_flights
            outbound_flights.update([create_flight(flight) for flight in timetable[0]])

            nonlocal inbound_flights
            inbound_flights.update([create_flight(flight) for flight in timetable[1]])

        if parameters.destination is None:
            def create_parameters_for_destination(destination: str) -> FlightSearchParameters:
                parameters_current_destination = copy.copy(parameters)
                parameters_current_destination.destination = destination
                return parameters_current_destination

            # route_map_wzz = self.wizzair.get_map()
            # if parameters.origin in route_map_wzz[1]:
            #     destinations_wzz = route_map_wzz[1][parameters.origin]
            #     for destination in destinations_wzz:
            #         # parameters_cur_dest = FlightSearchParameters()
            #         # parameters_cur_dest.origin = parameters.origin
            #         # parameters_cur_dest.outbound_departure_date_from = parameters.outbound_departure_date_from
            #         # parameters_cur_dest.outbound_departure_date_to = parameters.outbound_departure_date_to
            #         # parameters_cur_dest.destination = destination
            #         # parameters_cur_dest.inbound_departure_date_from = parameters.inbound_departure_date_from
            #         # parameters_cur_dest.inbound_departure_date_to = parameters.inbound_departure_date_to
            #         parameters_current_destination = create_parameters_for_destination(destination)
            #         search_wzz(parameters_wzz=parameters_current_destination)

            destinations_ryr = self.ryanair.get_destinations(parameters.origin)
            for destination in destinations_ryr:
                parameters_current_destination = create_parameters_for_destination(destination.code)
                search_ryr(parameters_ryr=parameters_current_destination)
        else:
            search_wzz(parameters_wzz=parameters)
            search_ryr(parameters_ryr=parameters)

        self.flights.update(outbound_flights)
        self.flights.update(inbound_flights)

        self.save_flights()

    def convert_currency(self, amount, from_currency, to_currency):
        c = CurrencyConverter()
        converted_amount = c.convert(amount, from_currency, to_currency)
        return converted_amount

    def get_flights(self, parameters: FlightSearchParameters) -> Set[Tuple[Flight, Flight]]:
        trips: Set[Tuple[Flight, Flight]] = set()

        for outbound_flight in self.flights:
            if outbound_flight.origin != parameters.origin:
                continue

            if outbound_flight.departure_date < parameters.outbound_departure_date_from or outbound_flight.departure_date > parameters.outbound_departure_date_to:
                continue

            if not parameters.destination is None and outbound_flight.destination != parameters.destination:
                continue

            for inbound_flight in self.flights:
                if inbound_flight.origin != outbound_flight.destination:
                    continue

                if inbound_flight.departure_date < outbound_flight.departure_date or inbound_flight.departure_date < parameters.inbound_departure_date_from or inbound_flight.departure_date > parameters.inbound_departure_date_to:
                    continue

                if inbound_flight.destination != parameters.origin:
                    continue

                if parameters.max_price is not None:
                    required_currency = parameters.max_price.currency

                    if outbound_flight.price.currency != required_currency:
                        outbound_flight.price = Price(value=self.convert_currency(amount=outbound_flight.price.value,
                                                                                  from_currency=outbound_flight.price.currency,
                                                                                  to_currency=required_currency),
                                                      currency=required_currency)

                    if inbound_flight.price.currency != required_currency:
                        inbound_flight.price = Price(value=self.convert_currency(amount=inbound_flight.price.value,
                                                                                  from_currency=inbound_flight.price.currency,
                                                                                  to_currency=required_currency),
                                                      currency=required_currency)

                    if outbound_flight.price.value + inbound_flight.price.value > parameters.max_price.value:
                        continue

                trip_duration = (inbound_flight.departure_date - outbound_flight.departure_date).days
                if trip_duration < parameters.min_trip_duration or trip_duration > parameters.max_trip_duration:
                    continue

                trips.add((outbound_flight, inbound_flight))

        return trips



if __name__ == '__main__':
    flight_db = FlightDB()
    # flight_db.load_flights()
    #
    # flight0 = Flight(origin="PRG", destination="BCN", departure_date="2024-05-05", arrival_date="2024-05-05", price=Price(currency="CZK", value=1999), airline="Wizzair")
    # flight_db.add(flight0)
    #
    # flight0 = Flight(origin="BCN", destination="PRG", departure_date="2024-05-15", arrival_date="2024-05-15",
    #                  price=Price(currency="CZK", value=1999), airline="Wizzair")
    # flight_db.add(flight0)
    #
    # flight_db.save_flights()

    # flight_db.fetch_airport_codes()
    flight_db.load_airport_codes()
    flight_db.fetch_airport_codes()
    flight_db.save_airport_codes()
