import asyncio
import logging
import os
from binance.client import Client
from database.database_connection import DatabaseConnection
from configuration_service import TypedConfigurationService
from strategy_factory import StrategyFactory
from exceptions import BaseTradingException

logger = logging.getLogger(__name__)

class SystemValidationError(BaseTradingException):
    """Raised when one or more system checks fail."""
    pass

async def check_database_connection() -> bool:
    try:
        db = DatabaseConnection()
        await db.connect()
        await db.disconnect()
        return True
    except BaseTradingException as exc:
        logger.error("Database check failed: %s", exc)
        return False

async def check_binance_connectivity() -> bool:
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET_KEY")
    if not api_key or not secret:
        logger.error("Missing Binance credentials")
        return False
    client = Client(api_key, secret, testnet=True)
    for attempt in range(3):
        try:
            await asyncio.wait_for(asyncio.to_thread(client.ping), timeout=5)
            return True
        except Exception as exc:
            logger.warning("Binance ping %s failed: %s", attempt + 1, exc)
            await asyncio.sleep(1)
    return False

def validate_configuration() -> bool:
    try:
        svc = TypedConfigurationService("config.json")
        svc.validate_required(["database_url"])
        return True
    except BaseTradingException as exc:
        logger.error("Config validation failed: %s", exc)
        return False

def test_strategy_creation() -> bool:
    try:
        StrategyFactory.create_strategy(
            "btc",
            Client("x", "y", testnet=True),
            "BTCUSDT",
            "15m",
        )
        return True
    except Exception as exc:
        logger.error("Strategy creation failed: %s", exc)
        return False

async def test_service_layer() -> bool:
    try:
        db = DatabaseConnection()
        await db.connect()
        await db.disconnect()
        return True
    except BaseTradingException as exc:
        logger.error("Service layer check failed: %s", exc)
        return False

async def validate_system() -> bool:
    """Validate all system components are working correctly."""
    checks = {
        'database': await check_database_connection(),
        'binance_api': await check_binance_connectivity(),
        'configuration': validate_configuration(),
        'strategies': test_strategy_creation(),
        'services': await test_service_layer()
    }

    failed_checks = [name for name, status in checks.items() if not status]
    if failed_checks:
        raise SystemValidationError(f"Failed checks: {failed_checks}")

    return True

async def main() -> None:
    if await validate_system():
        print("System health check passed.")

if __name__ == "__main__":
    asyncio.run(main())
