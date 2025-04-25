import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from trip_searcher.trip_searcher import TripSearcher


class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()

        self.trip_searcher = TripSearcher(train_route_parser=False)

        # Register handlers
        self.register_handlers()

    def register_handlers(self):
        @self.dp.message()
        async def cmd_start(message: Message):
            query = message.text
            response = self.trip_searcher.search(query)
            await message.answer(response)

    def run(self):
        print("Bot is running...")
        asyncio.run(self.dp.start_polling(self.bot))


if __name__ == "__main__":
    bot = TelegramBot(token="7999001822:AAHwCcbv_ptDNg0QHdgi7UUa8lSrYaiJT5U")
    bot.run()
