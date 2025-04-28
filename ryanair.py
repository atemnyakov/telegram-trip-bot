from __future__ import annotations
import requests
from typing import List, Dict, Tuple, Any, Optional


class Ryanair:
    class Coordinates:
        def __init__(self, latitude: float, longitude: float):
            self.latitude = latitude
            self.longitude = longitude

        def __repr__(self) -> str:
            return f"Coordinates(latitude={self.latitude}, longitude={self.longitude})"

    class City:
        def __init__(self, name: str, code: str):
            self.name = name
            self.code = code

        def __repr__(self) -> str:
            return f"City(name='{self.name}', code='{self.code}')"

    class Region:
        def __init__(self, name: str, code: str):
            self.name = name
            self.code = code

        def __repr__(self) -> str:
            return f"Region(name='{self.name}', code='{self.code}')"

    class Country:
        def __init__(
                self, code: str, iso3code: str, name: str, currency: str,
                default_airport_code: str, schengen: bool
        ):
            self.code = code
            self.iso3code = iso3code
            self.name = name
            self.currency = currency
            self.default_airport_code = default_airport_code
            self.schengen = schengen

        def __repr__(self) -> str:
            return (
                f"Country(code='{self.code}', iso3code='{self.iso3code}', name='{self.name}', "
                f"currency='{self.currency}', default_airport_code='{self.default_airport_code}', "
                f"schengen={self.schengen})"
            )

    class Airport:
        def __init__(
                self, code: str, name: str, seo_name: str, aliases: Any, base: str,
                city: Ryanair.City, region: Ryanair.Region, country: Ryanair.Country,
                coordinates: Ryanair.Coordinates, time_zone: str):
            self.code = code
            self.name = name
            self.seo_name = seo_name
            self.aliases = aliases
            self.base = base
            self.city = city
            self.region = region
            self.country = country
            self.coordinates = coordinates
            self.time_zone = time_zone

        @staticmethod
        def from_data(airport_data: Dict[str, Any]) -> Ryanair.Airport:
            city = Ryanair.City(airport_data['city']['name'], airport_data['city']['code'])
            region = Ryanair.Region(airport_data['region']['name'], airport_data['region']['code'])
            country = Ryanair.Country(
                airport_data['country']['code'],
                airport_data['country']['iso3code'],
                airport_data['country']['name'],
                airport_data['country']['currency'],
                airport_data['country']['defaultAirportCode'],
                airport_data['country']['schengen']
            )
            coordinates = Ryanair.Coordinates(
                airport_data['coordinates']['latitude'],
                airport_data['coordinates']['longitude']
            )

            return Ryanair.Airport(
                airport_data['code'],
                airport_data['name'],
                airport_data['seoName'],
                airport_data['aliases'],
                airport_data['base'],
                city,
                region,
                country,
                coordinates,
                airport_data['timeZone']
            )

        def __repr__(self) -> str:
            return f"Airport(name='{self.name}', code='{self.code}', city={self.city}, country={self.country})"

        def __str__(self) -> str:
            return f"{self.name} ({self.code})"

    class Price:
        def __init__(self, currency: str, value: float):
            self.currency = currency
            self.value = value

        def __repr__(self) -> str:
            return f"RyanairAPI.Price(currency='{self.currency}', value={self.value})"

        def __str__(self) -> str:
            return f"{self.value} {self.currency}"

    class Flight:
        def __init__(
                self, origin: str, destination: str, departure_date: str,
                arrival_date: str, price: Ryanair.Price
        ):
            self.origin = origin
            self.destination = destination
            self.departure_date = departure_date
            self.arrival_date = arrival_date
            self.price = price

        def __repr__(self) -> str:
            return (
                f"RyanairAPI.Flight(origin='{self.origin}', destination='{self.destination}', "
                f"departure_date='{self.departure_date}', arrival_date='{self.arrival_date}', price={repr(self.price)})"
            )

        def __str__(self) -> str:
            return (
                f"From: {self.origin}, {self.departure_date} to: {self.destination}, "
                f"{self.arrival_date} for {self.price}"
            )

    def get_airports(self) -> Optional[List[Ryanair.Airport]]:
        # Construct the API URL
        url = f"https://www.ryanair.com/api/views/locate/5/airports/en/active"

        # Send the GET request to the API
        response = requests.get(url=url)

        # Log the request URL for debugging
        print("Request URL:", response.url)

        # Parse the response JSON only if the request was successful (status code 200)
        if response.status_code != 200:
            return None

        airports: List[Ryanair.Airport] = []

        data = response.json()

        for airport_data in data:
            airport = Ryanair.Airport.from_data(airport_data)
            airports.append(airport)

        return airports

    def get_airport_info(self, airport_code: str) -> Optional[Ryanair.Airport]:
        """
        Fetches airport information from the Ryanair API.

        Args:
            airport_code (str): The IATA airport code (e.g., "DUB" for Dublin Airport).

        Returns:
            Optional[Ryanair.Airport]: An Airport object if the request is successful, otherwise None.
        """
        # Construct the API URL
        url = f"https://www.ryanair.com/api/views/locate/5/airports/en/{airport_code}"

        # Send the GET request to the API
        response = requests.get(url=url)

        # Log the request URL for debugging
        print("Request URL:", response.url)

        # Parse the response JSON only if the request was successful (status code 200)
        if response.status_code != 200:
            return None

        airport_data = response.json()

        # Convert API response into an Airport object
        return self.Airport.from_data(airport_data)

    def get_destinations(self, origin: str) -> List[Ryanair.Airport]:
        """
        Fetches a list of destination airports for a given origin airport from the Ryanair API.

        Args:
            origin (str): The IATA code of the origin airport (e.g., "DUB" for Dublin Airport).

        Returns:
            List[Ryanair.Airport]: A list of Airport objects representing possible destinations.
            Returns an empty list if the request fails or no destinations are found.
        """
        # Construct the API URL
        url = f"https://www.ryanair.com/api/views/locate/searchWidget/routes/en/airport/{origin}"

        # Send the GET request to the API
        response = requests.get(url=url)

        # Log the request URL for debugging
        print("Request URL:", response.url)

        # Parse the response JSON only if the request was successful (status code 200)
        if response.status_code != 200:
            return []

        destinations_data = response.json()

        # Convert API response into a list of Airport objects
        return [
            self.Airport.from_data(destination_data['arrivalAirport'])
            for destination_data in destinations_data
        ] if destinations_data else []

    import requests
    from typing import List

    def get_cheapest_flights(self, origin: str, destination: str, departure_date: str, currency: str = "EUR") -> List[
        Ryanair.Flight]:
        """
        Fetches the cheapest available flights from the Ryanair API for a given route and date.

        Args:
            origin (str): The IATA code of the origin airport (e.g., "DUB").
            destination (str): The IATA code of the destination airport (e.g., "STN").
            departure_date (str): The departure date in YYYY-MM format.
            currency (str, optional): The currency code (default is "EUR").

        Returns:
            List[Ryanair.Flight]: A list of available flights with prices. Returns an empty list if no flights are found.
        """
        # Construct the API URL
        url = f"https://www.ryanair.com/api/farfnd/v4/oneWayFares/{origin}/{destination}/cheapestPerDay"

        # Set request parameters
        params = {
            "outboundMonthOfDate": departure_date,
            "currency": currency
        }

        # Send the GET request to the API
        response = requests.get(url=url, params=params)

        # Log the request URL for debugging
        print("Request URL:", response.url)

        # Parse response JSON only if the request was successful
        if response.status_code != 200:
            return []

        flights_data = response.json()

        # Extract outbound fares, ensuring keys exist before accessing them
        fares = flights_data.get("outbound", {}).get("fares", [])

        # Convert API response into a list of Flight objects, filtering out unavailable flights
        return [
            self.Flight(
                origin,
                destination,
                fare["departureDate"],
                fare["arrivalDate"],
                self.Price(fare["price"]["currencyCode"], fare["price"]["value"])
            )
            for fare in fares if not fare.get("unavailable", False)
        ]

    def get_one_way_fares(
            self,
            origin: str,
            outbound_departure_date_from: str,
            outbound_departure_date_to: str,
            outbound_days_of_week: List[str] = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
            currency: str = "EUR",
            market: str = "en-gb",
            adult_pax_count: int = 1,
    ) -> List[Ryanair.Flight]:
        flights = []

        """
        Fetches available one-way flights from the Ryanair API for the given origin and date range.

        Args:
            origin (str): IATA code of the departure airport (e.g., "DUB").
            outbound_departure_date_from (str): Earliest outbound departure date (YYYY-MM-DD).
            outbound_departure_date_to (str): Latest outbound departure date (YYYY-MM-DD).

        Returns:
            List[RyanairAPI.Flight]:
            A list of outbound flights.
        """
        # Construct the API URL
        url = f"https://www.ryanair.com/api/farfnd/v4/oneWayFares"

        # Set request parameters
        params = {
            "departureAirportIataCode": origin,
            "outboundDepartureDateFrom": outbound_departure_date_from,
            "outboundDepartureDateTo": outbound_departure_date_to,
            "outboundDepartureDaysOfWeek": outbound_days_of_week,
            "market": market,
            "adultPaxCount": adult_pax_count,
            "currency": currency,
            "searchMode": 'ALL'
        }

        # Send the GET request to the API
        response = requests.get(url=url, params=params)

        # Log the request URL for debugging
        print("Request URL:", response.url)

        # Parse the response JSON only if the request was successful
        if response.status_code != 200:
            return []

        trips_data = response.json()

        def process_flight(flight_data: Optional[dict]) -> Optional[Ryanair.Flight]:
            """
            Converts flight data from the API response into a Flight object.

            Args:
                flight_data (Optional[dict]): The raw flight data from the API.

            Returns:
                Optional[Ryanair.Flight]: A Flight object if valid data is present, otherwise None.
            """
            if not flight_data:
                return None

            return Ryanair.Flight(
                origin=flight_data["departureAirport"]["iataCode"],
                destination=flight_data["arrivalAirport"]["iataCode"],
                departure_date=flight_data["departureDate"],
                arrival_date=flight_data["arrivalDate"],
                price=Ryanair.Price(
                    currency=flight_data["price"]["currencyCode"],
                    value=flight_data["price"]["value"]
                )
            )

        # Extract flight fares if available
        for trip_data in trips_data.get("fares", []):
            outbound_flight = process_flight(trip_data.get("outbound"))

            # Store the flight pair as a tuple
            flights.append(outbound_flight)

        return flights

    def get_round_trip_fares(
            self,
            origin: str,
            outbound_departure_date_from: str,
            outbound_departure_date_to: str,
            inbound_departure_date_from: str,
            inbound_departure_date_to: str,
            outbound_days_of_week: List[str] = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
            duration_from: int = 2,
            duration_to: int = 7,
            currency: str = "EUR",
            market: str = "en-gb",
            adult_pax_count: int = 1,
    ) -> List[Tuple[Optional[Ryanair.Flight], Optional[Ryanair.Flight]]]:
        """
        Fetches available round-trip flights from the Ryanair API for the given origin and date range.

        Args:
            origin (str): IATA code of the departure airport (e.g., "DUB").
            outbound_departure_date_from (str): Earliest outbound departure date (YYYY-MM-DD).
            outbound_departure_date_to (str): Latest outbound departure date (YYYY-MM-DD).
            inbound_departure_date_from (str): Earliest inbound departure date (YYYY-MM-DD).
            inbound_departure_date_to (str): Latest inbound departure date (YYYY-MM-DD).

        Returns:
            List[Tuple[Optional[Ryanair.Flight], Optional[Ryanair.Flight]]]:
            A list of tuples containing outbound and inbound flights.
            If a flight is unavailable, the corresponding tuple element is None.
        """
        flights = []

        # Construct the API URL and request parameters
        url = f"https://www.ryanair.com/api/farfnd/v4/roundTripFares"

        params = {
            "departureAirportIataCode": origin,
            "outboundDepartureDateFrom": outbound_departure_date_from,
            "outboundDepartureDateTo": outbound_departure_date_to,
            "inboundDepartureDateFrom": inbound_departure_date_from,
            "inboundDepartureDateTo": inbound_departure_date_to,
            "outbound_days_of_week": outbound_days_of_week,
            "durationFrom": duration_from,
            "durationTo": duration_to,
            "currency": currency,
            "market": market,
            "adultPaxCount": adult_pax_count,
            "searchMode": 'ALL'
        }

        # Send the GET request to the API
        response = requests.get(url=url, params=params)

        # Log the request URL for debugging
        print("Request URL:", response.url)

        # Parse the response JSON only if the request was successful
        if response.status_code != 200:
            return []

        trips_data = response.json()

        def process_flight(flight_data: Optional[dict]) -> Optional[Ryanair.Flight]:
            """
            Converts flight data from the API response into a Flight object.

            Args:
                flight_data (Optional[dict]): The raw flight data from the API.

            Returns:
                Optional[Ryanair.Flight]: A Flight object if valid data is present, otherwise None.
            """
            if not flight_data:
                return None

            return Ryanair.Flight(
                origin=flight_data["departureAirport"]["iataCode"],
                destination=flight_data["arrivalAirport"]["iataCode"],
                departure_date=flight_data["departureDate"],
                arrival_date=flight_data["arrivalDate"],
                price=Ryanair.Price(
                    currency=flight_data["price"]["currencyCode"],
                    value=flight_data["price"]["value"]
                )
            )

        # Extract flight fares if available
        for trip_data in trips_data.get("fares", []):
            outbound_flight = process_flight(trip_data.get("outbound"))
            return_flight = process_flight(trip_data.get("inbound"))

            # Store the flight pair as a tuple
            flights.append((outbound_flight, return_flight))

        return flights

    def get_flights(self,
                    origin: str,
                    destination: str,
                    departure_date: str,
                    return_date: str = None,
                    flex_days_before_departure: int = 4,
                    flex_days_after_departure: int = 2,
                    flex_days_before_return: int = 4,
                    flex_days_after_return: int = 2,
                    market: str = "en-gb") -> Tuple[List[Flight], Optional[List[Flight]]]:

        """ Retrieves available flights from the Ryanair API based on the provided parameters. """

        url = f"https://www.ryanair.com/api/booking/v4/{market}/availability"

        params = {
            "ADT": 1,
            "CHD": 0,
            "INF": 0,
            "TEEN": 0,  # Passenger counts
            "Origin": origin,
            "Destination": destination,
            "DateOut": departure_date,
            "DateIn": return_date or "",
            "FlexDaysBeforeOut": flex_days_before_departure,
            "FlexDaysOut": flex_days_after_departure,
            "FlexDaysBeforeIn": flex_days_before_return,
            "FlexDaysIn": flex_days_after_return,
            "RoundTrip": bool(return_date),
            "ToUs": "AGREED",
            "promoCode": "",
            "Disc": 0,
            "IncludeConnectingFlights": "false"
        }

        response = requests.get(url=url, params=params)
        print("Request URL:", response.url)  # Debugging

        if response.status_code != 200:
            return [], None

        availability_data = response.json()
        trips = availability_data.get('trips', [])

        def parse_flights(trip):
            """Extracts flight data from a trip object."""
            return [
                Ryanair.Flight(
                    origin=trip['origin'], destination=trip['destination'],
                    departure_date=flight['time'][0], arrival_date=flight['time'][1],
                    price=Ryanair.Price(currency=flight['regularFare']['fares'][0]['amount'],
                                        value=availability_data['currency'])
                )
                for date in trip['dates'] for flight in date['flights'] if flight['faresLeft'] >= 1
            ]

        outbound_flights = parse_flights(trips[0]) if len(trips) > 0 else []
        inbound_flights = parse_flights(trips[1]) if len(trips) > 1 else None

        return outbound_flights, inbound_flights

if __name__ == '__main__':
    ryanair = Ryanair()

    airport_info = ryanair.get_airport_info('PRG')
    print(airport_info)

    destinations = ryanair.get_destinations(origin='PRG')
    print(destinations)

    cheapest_flights = ryanair.get_cheapest_flights(
        origin='PRG',
        destination='BGY',
        departure_date='2025-06-06'
    )
    print(cheapest_flights)

    one_way_fares = ryanair.get_one_way_fares(
        origin='PRG',
        outbound_departure_date_from='2025-06-06',
        outbound_departure_date_to='2025-06-06'
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
        destination='BGY',
        departure_date='2025-06-06'
    )
    print(one_way_flights)

    round_flights = ryanair.get_flights(
        origin='PRG',
        destination='BGY',
        departure_date='2025-06-01',
        flex_days_before_departure=0,
        flex_days_after_departure=5,
        return_date='2025-06-01',
        flex_days_before_return=0,
        flex_days_after_return=5
    )
    print(round_flights[0])
    print(round_flights[1])

