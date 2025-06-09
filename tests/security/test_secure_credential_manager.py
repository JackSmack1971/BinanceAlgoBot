import asyncio
import base64
import os
import pytest
from cryptography.fernet import Fernet

from src.security import SecureCredentialManager, TradingCredentials, CredentialError


@pytest.fixture()
def encryption_key() -> str:
    return Fernet.generate_key().decode()


@pytest.mark.asyncio
async def test_get_credentials(monkeypatch, encryption_key):
    manager = SecureCredentialManager(encryption_key)
    cipher = Fernet(encryption_key.encode())
    api = cipher.encrypt(b"key123456")
    secret = cipher.encrypt(b"secret123456")
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", base64.b64encode(api).decode())
    monkeypatch.setenv("BINANCE_TESTNET_SECRET_KEY", base64.b64encode(secret).decode())
    creds = await manager.get_credentials("BINANCE", "testnet")
    assert creds.api_key == "key123456"
    assert creds.environment == "testnet"
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY")
    monkeypatch.delenv("BINANCE_TESTNET_SECRET_KEY")


@pytest.mark.asyncio
async def test_rotate_credentials(monkeypatch, encryption_key):
    manager = SecureCredentialManager(encryption_key)
    new_creds = TradingCredentials(api_key="newkey1234", api_secret="newsecret1234", environment="production")
    await manager.rotate_credentials("BINANCE", new_creds)
    env_api = os.environ["BINANCE_PRODUCTION_API_KEY"]
    cipher = Fernet(encryption_key.encode())
    assert cipher.decrypt(base64.b64decode(env_api.encode())).decode() == "newkey1234"
    monkeypatch.delenv("BINANCE_PRODUCTION_API_KEY")
    monkeypatch.delenv("BINANCE_PRODUCTION_SECRET_KEY")


@pytest.mark.asyncio
async def test_invalid_credentials(encryption_key):
    manager = SecureCredentialManager(encryption_key)
    with pytest.raises(CredentialError):
        await manager.get_credentials("BINANCE", "testnet")
