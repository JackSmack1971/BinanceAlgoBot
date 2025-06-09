import asyncio
from binance import Client
from configuration_service import ConfigurationService

class BinanceExchangeInterface:
    """Lightweight wrapper around the Binance API client."""

    def __init__(self, config_service: ConfigurationService | None = None):
        """Initialize the API client using values from ``config_service``."""
        self.config_service = config_service or ConfigurationService()
        self.api_key = self.config_service.get_config('api_key')
        self.api_secret = self.config_service.get_config('secret_key')
        self.config_service.validate_required(['api_key', 'secret_key'])
        self.client = Client(
            self.api_key,
            self.api_secret,
            loop=asyncio.get_event_loop(),
        )
