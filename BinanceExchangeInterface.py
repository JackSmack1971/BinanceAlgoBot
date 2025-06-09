import asyncio
import os
from binance import Client
from configuration_service import ConfigurationService
from security import SecureCredentialManager, CredentialError

class BinanceExchangeInterface:
    """Lightweight wrapper around the Binance API client."""

    def __init__(self, config_service: ConfigurationService | None = None):
        """Initialize the API client using values from ``config_service``."""
        self.config_service = config_service or ConfigurationService()
        enc_key = os.getenv("CREDENTIAL_ENCRYPTION_KEY", "")
        if not enc_key:
            raise CredentialError("Missing encryption key")
        manager = SecureCredentialManager(enc_key)
        creds = asyncio.run(manager.get_credentials("BINANCE", "production"))
        self.client = Client(
            creds.api_key,
            creds.api_secret,
            loop=asyncio.get_event_loop(),
        )
