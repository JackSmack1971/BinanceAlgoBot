import os
import pytest
from configuration_service import TypedConfigurationService
from exceptions import ConfigurationError, BaseTradingException


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "env_key")
    service = TypedConfigurationService('config.json')
    service.declare_config('api_key', str, '')
    assert service.get_config('api_key') == 'env_key'
    monkeypatch.delenv("BINANCE_API_KEY")


def test_validate_required(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_SECRET_KEY", "s")
    service = TypedConfigurationService('config.json')
    service.declare_config('api_key', str, '')
    service.declare_config('secret_key', str, '')
    service.validate_required(['api_key', 'secret_key'])
    monkeypatch.delenv("BINANCE_API_KEY")
    monkeypatch.delenv("BINANCE_SECRET_KEY")


def test_validate_required_failure(monkeypatch):
    service = TypedConfigurationService('config.json')
    service.declare_config('api_key', str, '')
    service.declare_config('secret_key', str, '')
    with pytest.raises(BaseTradingException):
        service.validate_required(['api_key', 'secret_key'])
