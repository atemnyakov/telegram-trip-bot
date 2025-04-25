from __future__ import annotations
import requests
from typing import List, Dict, Tuple, Optional
from itertools import zip_longest


class Wizzair:
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

        def __init__(self, code, name, currency):
            self.code = code
            self.name = name
            self.currency = currency

        def __str__(self):
            return f"{self.name} ({self.code}), currency: {self.currency}"

        def __repr__(self):
            return f"Country(code={self.code!r}, name={self.name!r}, currency={self.currency!r})"

    class Destination:
        """
        Represents a travel destination with a unique code, name, optional aliases,
        associated country, and geographical coordinates.
        """

        def __init__(self, code, name, aliases, country, coordinates):
            self.code = code
            self.name = name
            self.aliases = aliases
            self.country = country
            self.coordinates = coordinates

        def __str__(self):
            aliases_str = ", ".join(self.aliases) if self.aliases else "None"
            return f"{self.name} ({self.code}), country: {self.country}, coordinates: {self.coordinates}"

        def __repr__(self):
            return (f"Destination(code={self.code!r}, name={self.name!r}, aliases={self.aliases!r}, "
                    f"country={self.country!r}, coordinates={self.coordinates!r})")

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

    class Flight:
        """
        Represents a flight with an origin, destination, departure date, and price.
        """

        def __init__(self, origin, destination, departure_date, price):
            self.origin = origin
            self.destination = destination
            self.departure_date = departure_date
            self.price = price

        def __str__(self):
            return f"From {self.origin} to {self.destination}, departure: {self.departure_date}, price: {self.price}"

        def __repr__(self):
            return (f"Flight(origin={self.origin!r}, destination={self.destination!r}, "
                    f"departure_date={self.departure_date!r}, price={self.price!r})")

    def __init__(self):
        """
        Initializes the class by setting up HTTP headers and retrieving
        the latest build number from Wizz Air's website.
        """

        # Define the default headers for HTTP requests
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://wizzair.com",
            "Referer": "https://wizzair.com",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            ),
        }

        # URL to fetch the current build number
        url = "https://www.wizzair.com/buildnumber"

        # Make the request with additional headers
        response = requests.get(
            url,
            headers={
                **self.headers,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                          "image/avif,image/webp,image/apng,*/*;q=0.8,"
                          "application/signed-exchange;v=b3;q=0.7"
            }
        )

        # Extract the build number if the request was successful
        if response.status_code == 200:
            self.build_number = response.text.split("/")[-1].strip("'")
        else:
            self.build_number = ""  # Default to an empty string if the request fails

    # def get_cheapest_flights(self, departure_station: str, months: int, discounted_only: bool = False):
    #     url = f'https://be.wizzair.com/{self.build_number}/Api/search/CheapFlights'
    #     payload = {
    #         "departureStation": departure_station,
    #         "months": months,
    #         "discountedOnly": discounted_only,
    #     }
    #     response = requests.post(url, headers=self.headers, json=payload)
    #
    #     if response.status_code == 200:
    #         return response.json().get("items", [])
    #     else:
    #         return None

    from typing import Tuple, List, Dict

    def get_map(self) -> Tuple[List[Destination], Dict[str, List[str]]]:
        """
        Fetches the map data from the Wizz Air API, extracting available destinations
        and their connections.

        Returns:
            Tuple[List[Destination], Dict[str, List[str]]]:
                - A list of Destination objects.
                - A dictionary mapping each destination code to a list of connected destination codes.
        """

        destinations = []  # List to store destination objects
        connections = {}  # Dictionary to store flight connections

        # Construct the API URL using the retrieved build number
        url = f"https://be.wizzair.com/{self.build_number}/Api/asset/map?languageCode=en-gb"

        # Send a GET request to the API
        response = requests.get(url, headers=self.headers)

        # Process the response if the request was successful
        if response.status_code == 200:
            map_data = response.json()
            cities_data = map_data.get("cities", [])

            for city_data in cities_data:
                # Extract coordinates
                longitude = city_data["longitude"]
                latitude = city_data["latitude"]
                coordinates = self.Coordinates(longitude=longitude, latitude=latitude)

                # Extract country details
                country = self.Country(
                    code=city_data["countryCode"],
                    name=city_data["countryName"],
                    currency=city_data["currencyCode"]
                )

                # Create a Destination object
                destination = self.Destination(
                    code=city_data["iata"],
                    name=city_data["shortName"],
                    aliases=city_data["aliases"],
                    country=country,
                    coordinates=coordinates
                )

                destinations.append(destination)

                # Extract and store connections (list of IATA codes)
                connections[destination.code] = [
                    connection["iata"] for connection in city_data.get("connections", [])
                ]

        return destinations, connections

    def get_farechart(self,
                      origin: str,
                      destination: str,
                      departure_date: str,
                      return_date: str,
                      day_interval: int = 7,
                      adult_count: int = 1,
                      wizzair_discount_club: bool = False) -> List[Tuple[Optional[Flight], Optional[Flight]]]:
        """
        Fetches fare chart data for flights between the given origin and destination.

        Args:
            origin (str): IATA code of the departure airport.
            destination (str): IATA code of the arrival airport.
            departure_date (str): Date of departure (YYYY-MM-DD format).
            return_date (str): Date of return (YYYY-MM-DD format).
            day_interval (int, optional): Number of days to consider around the given dates. Defaults to 7.
            adult_count (int, optional): Number of adult passengers. Defaults to 1.
            wizzair_discount_club (bool, optional): Whether to use Wizz Air Discount Club fares. Defaults to False.

        Returns:
            List[Tuple[Optional[Flight], Optional[Flight]]]: A list of tuples containing outbound and return flights.
            If a flight is not available, it will be `None` in the tuple.
        """

        flights = []  # List to store (outbound, return) flight pairs

        # Construct the API endpoint using the retrieved build number
        url = f"https://be.wizzair.com/{self.build_number}/Api/asset/farechart"

        # Prepare the payload for the request
        payload = {
            "isRescueFare": False,
            "adultCount": adult_count,
            "childCount": 0,
            "dayInterval": day_interval,
            "wdc": wizzair_discount_club,
            "isFlightChange": False,
            "flightList": [
                {
                    "departureStation": origin,
                    "arrivalStation": destination,
                    "date": departure_date
                },
                {
                    "departureStation": destination,
                    "arrivalStation": origin,
                    "date": return_date
                }
            ]
        }

        # Send a POST request to the API
        response = requests.post(url, headers=self.headers, json=payload)

        # Parse the response JSON if the request was successful
        data = response.json() if response.status_code == 200 else None

        def process_flights(flight_data_list: Optional[List[dict]]) -> Optional[List[Optional[Flight]]]:
            """
            Converts API flight data into a list of Flight objects.

            Args:
                flight_data_list (Optional[List[dict]]): List of flight data dictionaries.

            Returns:
                Optional[List[Optional[Flight]]]: A list of Flight objects or None if data is missing.
            """
            if flight_data_list is None:
                return None

            return [
                self.Flight(
                    origin=flight_data["departureStation"],
                    destination=flight_data["arrivalStation"],
                    departure_date=flight_data["date"],
                    price=self.Price(
                        currency=flight_data["price"]["currencyCode"],
                        value=flight_data["price"]["amount"]
                    )
                ) if flight_data else None
                for flight_data in flight_data_list
            ]

        if data:
            outbound_flights = process_flights(data.get("outboundFlights"))
            return_flights = process_flights(data.get("returnFlights"))

            # Pair outbound and return flights together
            flights = list(zip_longest(
                outbound_flights if outbound_flights else [],
                return_flights if return_flights else [],
                fillvalue=None
            ))

        return flights

    def get_flights(self,
                    origin: str,
                    destination: Optional[str] = None,
                    departure_date: Optional[str] = None,
                    return_date: Optional[str] = None,
                    trip_duration: Optional[str] = None,
                    months: Optional[List[str]] = None,
                    passengers_count: int = 1) -> List[Tuple[Optional[Flight], Optional[Flight]]]:
        """
        Fetches flight options from the Wizz Air API based on the given parameters.

        Args:
            origin (str): IATA code of the departure airport.
            destination (Optional[str]): IATA code of the arrival airport. Defaults to None.
            departure_date (Optional[str]): Exact departure date (YYYY-MM-DD). Defaults to None.
            return_date (Optional[str]): Exact return date (YYYY-MM-DD). Defaults to None.
            trip_duration (Optional[str]): Duration of the trip in days. Defaults to None.
            months (Optional[List[str]]): List of months for flexible searches. Defaults to None.
            passengers_count (int, optional): Number of passengers. Defaults to 1.

        Returns:
            List[Tuple[Optional[Flight], Optional[Flight]]]: A list of outbound and return flight pairs.
            If a flight is unavailable, its value will be `None` in the tuple.
        """

        flights = []  # List to store (outbound, return) flight pairs

        # Construct API endpoint
        url = f"https://be.wizzair.com/{self.build_number}/Api/search/SmartSearchCheapFlights"

        # Prepare request payload
        payload = {
            "arrivalStations": [destination] if destination else None,
            "departureStations": [origin],
            "tripDuration": trip_duration,
            "isReturnFlight": bool(return_date or trip_duration),
            "stdPlan": months,
            "pax": passengers_count,
            "dateFilterType": "Exact" if (departure_date and return_date) or (
                        departure_date and not trip_duration) else "Flexible",
            "departureDate": departure_date,
            "returnDate": return_date
        }

        # Send request to API
        response = requests.post(url, headers=self.headers, json=payload)

        # Check if the request was successful
        data = response.json() if response.status_code == 200 else None

        def process_flight(flight_data: Optional[dict]) -> Optional[Flight]:
            """
            Converts flight data into a Flight object.

            Args:
                flight_data (Optional[dict]): Dictionary containing flight details.

            Returns:
                Optional[Flight]: A Flight object if data is available, otherwise None.
            """
            if not flight_data:
                return None

            return self.Flight(
                origin=flight_data["departureStation"],
                destination=flight_data["arrivalStation"],
                departure_date=flight_data["std"],
                price=self.Price(
                    currency=flight_data["regularPrice"]["currencyCode"],
                    value=flight_data["regularPrice"]["amount"]
                )
            )

        # Process API response
        if data and "items" in data:
            for trip_data in data["items"]:
                outbound_flight = process_flight(trip_data.get("outboundFlight"))
                return_flight = process_flight(trip_data.get("inboundFlight"))

                # Append flight pair (outbound, return)
                flights.append((outbound_flight, return_flight))

        return flights

    def get_timetable(self,
                      origin: str,
                      destination: str,
                      outbound_departure_date_from: str,
                      outbound_departure_date_to: str,
                      inbound_departure_date_from: str,
                      inbound_departure_date_to: str,
                      passengers_count: int = 1,
                      price_type: str = 'regular') -> Tuple[List[Optional[Flight]], List[Optional[Flight]]]:
        """
        Fetches the flight timetable from the Wizz Air API based on the provided date range.

        Args:
            origin (str): IATA code of the departure airport.
            destination (str): IATA code of the arrival airport.
            outbound_departure_date_from (str): Start date for outbound flights (YYYY-MM-DD).
            outbound_departure_date_to (str): End date for outbound flights (YYYY-MM-DD).
            inbound_departure_date_from (str): Start date for inbound flights (YYYY-MM-DD).
            inbound_departure_date_to (str): End date for inbound flights (YYYY-MM-DD).
            passengers_count (int, optional): Number of adult passengers. Defaults to 1.
            price_type (str, optional): Type of price ("regular" or "discount"). Defaults to "regular".

        Returns:
            Tuple[List[Optional[Flight]], List[Optional[Flight]]]:
            - List of outbound flights.
            - List of inbound flights.
            If no flights are found, lists may contain `None` values.
        """

        # Construct API endpoint
        url = f"https://be.wizzair.com/{self.build_number}/Api/search/timetable"

        # Prepare request payload
        payload = {
            "flightList": [
                {
                    "departureStation": origin,
                    "arrivalStation": destination,
                    "from": outbound_departure_date_from,
                    "to": outbound_departure_date_to
                },
                {
                    "departureStation": destination,
                    "arrivalStation": origin,
                    "from": inbound_departure_date_from,
                    "to": inbound_departure_date_to
                }
            ],
            "priceType": price_type,
            "adultCount": passengers_count,
            "childCount": 0,
            "infantCount": 0
        }

        # Send request to API
        response = requests.post(url, headers=self.headers, json=payload)

        # Check if the request was successful
        data = response.json() if response.status_code == 200 else None

        def process_flights(flight_data_list: Optional[List[dict]]) -> List[Optional[Flight]]:
            """
            Converts a list of flight data dictionaries into Flight objects.

            Args:
                flight_data_list (Optional[List[dict]]): List of flight data.

            Returns:
                List[Optional[Flight]]: List of Flight objects, with `None` for missing data.
            """
            if not flight_data_list:
                return []

            return [
                self.Flight(
                    origin=flight_data["departureStation"],
                    destination=flight_data["arrivalStation"],
                    departure_date=flight_data["departureDate"],
                    price=self.Price(
                        currency=flight_data["price"]["currencyCode"],
                        value=flight_data["price"]["amount"]
                    )
                ) if flight_data else None
                for flight_data in flight_data_list
            ]

        # Process API response
        outbound_flights = process_flights(data.get("outboundFlights")) if data else []
        inbound_flights = process_flights(data.get("returnFlights")) if data else []

        return outbound_flights, inbound_flights
