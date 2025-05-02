import asyncio
from datetime import datetime

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from trip_searcher.trip_searcher import TripSearcher


class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()

        self.trip_searcher = TripSearcher(train_route_parser=False, train_price_parser=False, train_date_classifier=False)

        # Register handlers
        self.register_handlers()

    def register_handlers(self):
        @self.dp.message()
        async def cmd_start(message: Message):
            query = message.text

            round_trips = self.trip_searcher.search(query)

            response = ""

            max_number_of_trips = 15

            number_of_found_trips = len(round_trips)

            if len(round_trips) <= max_number_of_trips:
                response += f"Я нашёл для тебя {number_of_found_trips} вариантов! Вот они:\n\n\n"
            else:
                response += f"Я нашёл для тебя {number_of_found_trips} вариантов, но из-за ограничения Телеграма могу показать только {max_number_of_trips}. Вот самые дешёвые из найдённых:\n\n\n"

            for i in range(min(number_of_found_trips, max_number_of_trips)):
                outbound_flight = round_trips[i][0]
                inbound_flight = round_trips[i][1]

                response += f"Маршрут туда: {outbound_flight.origin} - {outbound_flight.destination}\nАвиакомпания: {outbound_flight.airline}\nДата: {outbound_flight.departure_date.date()}\nЦена: {outbound_flight.price}\n\n"
                response += f"Маршрут назад: {inbound_flight.origin} - {inbound_flight.destination}\nАвиакомпания: {inbound_flight.airline}\nДата: {inbound_flight.departure_date.date()}\nЦена: {inbound_flight.price}\n\n\n"

            await message.answer(response)

    async def remove_webhook(self):
        # Delete webhook before starting polling
        await self.bot.delete_webhook()
        print("Webhook deleted successfully!")

    def run(self):
        print("Bot is running...")
        # asyncio.run(self.remove_webhook())
        asyncio.run(self.dp.start_polling(self.bot))


if __name__ == "__main__":
    bot = TelegramBot(token="7999001822:AAGxIGE9t-I207pcOnOieITv9PBaggTvYGc")
    bot.run()
