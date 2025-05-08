class ServiceLocator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.services = {}
        return cls._instance

    def register(self, service_name, service_instance):
        self.services[service_name] = service_instance

    def get(self, service_name):
        try:
            return self.services[service_name]
        except KeyError:
            raise ConfigurationError(f"Service {service_name} not registered.")

# Example usage:
# service_locator = ServiceLocator()
# service_locator.register("PerformanceMetricsService", PerformanceMetricsServiceImpl())
# performance_metrics_service = service_locator.get("PerformanceMetricsService")