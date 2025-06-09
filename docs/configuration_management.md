# Configuration Management

The project uses a combination of `config.json`, `config.py`, and `configuration_service.py` for managing configuration parameters.

## Configuration Files

*   `config.json`: This file stores various configuration parameters in JSON format.
*   `config.py`: This file defines functions for accessing configuration parameters from `config.json` and environment variables.

## Configuration Service

*   `configuration_service.py`: This file provides a service for managing and accessing configuration parameters. It loads the configuration from `config.json` and allows access to the parameters through getter methods.

## How Parameters are Loaded and Accessed

1.  The `configuration_service.py` loads the configuration from `config.json` when the service is initialized.
2.  The `config.py` provides the `get_config()` function to access configuration parameters. This function first checks if the parameter is defined as an environment variable. If not, it retrieves the parameter from the loaded `config.json`.
3.  Components can then use the `get_config()` function to access the configuration parameters they need.

### Example

```python
from configuration_service import ConfigurationService

config = ConfigurationService()
api_key = config.get_config("api_key")
```
