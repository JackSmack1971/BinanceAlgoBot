import json
import os
from typing import List, Dict, Callable, Type, Any
from utils import handle_error
from exceptions import ConfigurationError
from validation import validate_risk

ENV_VAR_MAPPING = {
    "api_key": "BINANCE_API_KEY",
    "secret_key": "BINANCE_SECRET_KEY",
    "database_url": "DATABASE_URL",
}

class ConfigurationService:
    @handle_error
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.observers: List[Callable] = []
        self.load_config()

class TypedConfigurationService(ConfigurationService):
    def __init__(self, config_file: str = 'config.json'):
        super().__init__(config_file)
        self.config_types: Dict[str, Type] = {}
        self.default_values: Dict[str, Any] = {}
        self.validate_config()

    def declare_config(self, key: str, config_type: Type, default_value: Any):
        self.config_types[key] = config_type
        self.default_values[key] = default_value
        if key not in self.config:
            self.config[key] = default_value

    def get_config(self, key: str, default=None):
        if key not in self.config_types:
            raise ValueError(f"Configuration key '{key}' not declared.")
        
        value = super().get_config(key, self.default_values[key])

        if not isinstance(value, self.config_types[key]):
            try:
                value = self.config_types[key](value)
            except ValueError:
                raise ValueError(f"Configuration key '{key}' has invalid type. Expected {self.config_types[key]}, got {type(value)}.")
        
        return value

    @handle_error
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{self.config_file}' not found.")
            self.config = {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in configuration file '{self.config_file}'.")
            self.config = {}

    @handle_error
    def get_config(self, key: str, default=None):
        env_key = ENV_VAR_MAPPING.get(key, key.upper())
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        return self.config.get(key, default)

    @handle_error
    def set_config(self, key: str, value):
        self.config[key] = value
        self.notify_observers()

    @handle_error
    def register_observer(self, observer: Callable):
        self.observers.append(observer)

    @handle_error
    def unregister_observer(self, observer: Callable):
        self.observers.remove(observer)

    @handle_error
    def notify_observers(self):
        for observer in self.observers:
            observer()

    @handle_error
    def validate_required(self, keys: List[str]):
        missing = []
        for key in keys:
            value = self.get_config(key)
            if value is None or str(value).strip() == "":
                missing.append(key)
        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {', '.join(missing)}"
            )

    @handle_error
    def validate_config(self) -> None:
        for key, expected in self.config_types.items():
            value = self.get_config(key, self.default_values.get(key))
            if expected is float and 'risk' in key:
                validate_risk(float(value))
            elif not isinstance(value, expected):
                try:
                    expected(value)
                except Exception:
                    raise ConfigurationError(
                        f"Invalid type for {key}: expected {expected.__name__}"
                    )

if __name__ == '__main__':
    # Example usage
    config_service = ConfigurationService('config.json')

    def my_observer():
        print("Configuration changed!")

    config_service.register_observer(my_observer)

    print(f"Initial value of 'api_key': {config_service.get_config('api_key')}")

    config_service.set_config('api_key', 'new_api_key')

    print(f"New value of 'api_key': {config_service.get_config('api_key')}")