import json
from typing import List, Dict, Callable

class ConfigurationService:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config: Dict = {}
        self.observers: List[Callable] = []
        self.load_config()

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

    def get_config(self, key: str, default=None):
        return self.config.get(key, default)

    def set_config(self, key: str, value):
        self.config[key] = value
        self.notify_observers()

    def register_observer(self, observer: Callable):
        self.observers.append(observer)

    def unregister_observer(self, observer: Callable):
        self.observers.remove(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer()

if __name__ == '__main__':
    # Example usage
    config_service = ConfigurationService('config.json')

    def my_observer():
        print("Configuration changed!")

    config_service.register_observer(my_observer)

    print(f"Initial value of 'api_key': {config_service.get_config('api_key')}")

    config_service.set_config('api_key', 'new_api_key')

    print(f"New value of 'api_key': {config_service.get_config('api_key')}")