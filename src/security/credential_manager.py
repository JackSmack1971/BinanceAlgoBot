from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict

from cryptography.fernet import Fernet

from exceptions import CredentialError

logger = logging.getLogger(__name__)

MASK = "****"


def _mask(value: str) -> str:
    if len(value) <= 4:
        return MASK
    return f"{MASK}{value[-4:]}"


@dataclass
class TradingCredentials:
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    environment: str = "production"  # "production" or "testnet"

    def __post_init__(self) -> None:
        self.validate_credentials()

    def validate_credentials(self) -> None:
        """Validate API key format and permissions."""
        if self.environment not in {"production", "testnet"}:
            raise CredentialError(f"Invalid environment: {self.environment}")
        for name, value in {"api_key": self.api_key, "api_secret": self.api_secret}.items():
            if not re.fullmatch(r"[A-Za-z0-9]{8,64}", value or ""):
                raise CredentialError(f"Invalid {name} format")


class SecureCredentialManager:
    def __init__(self, encryption_key: str) -> None:
        self.cipher = Fernet(encryption_key.encode())
        self._store: Dict[str, Dict[str, bytes]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    async def get_credentials(self, exchange: str, environment: str) -> TradingCredentials:
        """Retrieve and decrypt credentials for specific exchange and environment."""
        key = f"{exchange}_{environment}"
        encrypted = self._store.get(key)
        if not encrypted:
            env_api = os.getenv(f"{exchange.upper()}_{environment.upper()}_API_KEY", "")
            env_secret = os.getenv(f"{exchange.upper()}_{environment.upper()}_SECRET_KEY", "")
            if not env_api or not env_secret:
                raise CredentialError("Credentials not found")
            encrypted = {
                "api": base64.b64decode(env_api.encode()),
                "secret": base64.b64decode(env_secret.encode()),
            }
            self._store[key] = encrypted
        api_key = self.cipher.decrypt(encrypted["api"]).decode()
        api_secret = self.cipher.decrypt(encrypted["secret"]).decode()
        creds = TradingCredentials(api_key=api_key, api_secret=api_secret, environment=environment)
        self.logger.info(
            "Loaded credentials for %s key=%s",
            environment,
            _mask(api_key),
        )
        return creds

    async def rotate_credentials(self, exchange: str, new_credentials: TradingCredentials) -> None:
        """Safely rotate API credentials with zero-downtime."""
        self.validate_permissions(new_credentials)
        key = f"{exchange}_{new_credentials.environment}"
        encrypted = {
            "api": self.cipher.encrypt(new_credentials.api_key.encode()),
            "secret": self.cipher.encrypt(new_credentials.api_secret.encode()),
        }
        self._store[key] = encrypted
        os.environ[f"{exchange.upper()}_{new_credentials.environment.upper()}_API_KEY"] = base64.b64encode(encrypted["api"]).decode()
        os.environ[f"{exchange.upper()}_{new_credentials.environment.upper()}_SECRET_KEY"] = base64.b64encode(encrypted["secret"]).decode()
        self.logger.info(
            "Rotated credentials for %s key=%s",
            new_credentials.environment,
            _mask(new_credentials.api_key),
        )

    def validate_permissions(self, credentials: TradingCredentials) -> Dict[str, bool]:
        """Test API permissions before use."""
        # Placeholder permission check. In production, make authenticated call
        has_trade = bool(credentials.api_key and credentials.api_secret)
        has_read = has_trade
        return {"read": has_read, "trade": has_trade}
