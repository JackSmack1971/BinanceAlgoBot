# Service Interfaces

This document describes the main service interfaces used throughout the project.

## MarketDataService
Retrieves historical and latest market data from the database. Implementations
must provide:

- `get_market_data(symbol: str) -> List[Dict[str, Any]]`
- `get_latest_market_data(symbol: str) -> Dict[str, Any]`

## IndicatorService
Handles calculation and storage of technical indicators.

- `calculate_indicators(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
- `get_indicators(market_data_id: int) -> List[Dict[str, Any]]`

## PerformanceMetricsService
Persists trade performance metrics and allows retrieval by trade id.

- `get_performance_metrics_by_trade_id(trade_id: int) -> Dict[str, Any]`
- `insert_performance_metrics(...) -> None`

## ServiceLocator
Simple dependency container that registers and retrieves service instances.
