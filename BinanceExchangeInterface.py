import asyncio
from binance import Client
from config import get_config

class BinanceExchangeInterface:
    def __init__(self):
        self.api_key = get_config('api_key')
        self.api_secret = get_config('secret_key')
        self.client = Client(self.api_key, self.api_secret, loop=asyncio.get_event_loop())