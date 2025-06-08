import os
from configuration_service import TypedConfigurationService


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("API_KEY", "env_key")
    service = TypedConfigurationService('config.json')
    service.declare_config('api_key', str, '')
    assert service.get_config('api_key') == 'env_key'
    monkeypatch.delenv("API_KEY")
