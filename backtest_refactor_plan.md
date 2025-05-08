# Backtest Refactor Plan

## Goal

Refactor `Backtester` and `BacktestExecutor` to clearly separate responsibilities between them, adhering to the Single Responsibility Principle.

## Analysis

### Current Responsibilities

**Backtester.py:**

*   Initializes the backtesting environment.
*   Takes a `Client` and `Strategy` as input.
*   Initializes `BacktestExecutor`, `PerformanceAnalyzer`, `ReportGenerator`, and `VisualizationService`.
*   `run()` method:
    *   Executes the backtest using `BacktestExecutor`.
    *   Analyzes performance using `PerformanceAnalyzer`.
    *   Returns the backtesting results.
*   `plot_results()` method:
    *   Plots the backtest results using `VisualizationService`.
*   `generate_report()` method:
    *   Generates a detailed report of the backtest results using `ReportGenerator`.
*   `compare_strategies()` method:
    *   Compares multiple backtesting results and generates a combined report.

**BacktestExecutor.py:**

*   Executes the backtest for a given strategy and historical data.
*   Takes a `Client` and `Strategy` as input.
*   `run()` method:
    *   Fetches historical data from Binance using the `Client`.
    *   Converts the data to a Pandas DataFrame.
    *   Applies the strategy to generate signals.
    *   Returns a DataFrame with backtesting results.

### Overlapping Functionality and Concerns

*   Both classes take a `Client` and `Strategy` as input.
*   Both classes have a `run()` method that executes the backtest.
*   `Backtester`'s `run()` method calls `BacktestExecutor`'s `run()` method, which seems redundant.
*   `Backtester` is responsible for both setting up the backtest and analyzing the results, which violates the Single Responsibility Principle.
*   `BacktestExecutor` is responsible for fetching historical data and applying the strategy, which could be separated further.

## Proposed Refactoring

1.  **Backtester:**
    *   Responsible for setting up and configuring the backtest.
    *   Takes a `BacktestConfiguration` object as input, which contains the `Client`, `Strategy`, start date, end date, and other configuration parameters.
    *   Initializes `BacktestExecutor`, `PerformanceAnalyzer`, `ReportGenerator`, and `VisualizationService`.
    *   `run()` method:
        *   Validates the configuration.
        *   Passes the configuration to `BacktestExecutor` to execute the backtest.
        *   Analyzes performance using `PerformanceAnalyzer`.
        *   Returns the backtesting results.
    *   `plot_results()` method:
        *   Plots the backtest results using `VisualizationService`.
    *   `generate_report()` method:
        *   Generates a detailed report of the backtest results using `ReportGenerator`.
    *   `compare_strategies()` method:
        *   Compares multiple backtesting results and generates a combined report.
2.  **BacktestExecutor:**
    *   Responsible for executing the backtest.
    *   Takes a `BacktestConfiguration` object as input.
    *   `run()` method:
        *   Fetches historical data from Binance using the `Client` in the `BacktestConfiguration`.
        *   Converts the data to a Pandas DataFrame.
        *   Applies the strategy to generate signals.
        *   Returns a `BacktestResult` object with backtesting results.
3.  **BacktestHelper:**
    *   A new helper class or module that contains the overlapping functionality.
    *   `fetch_historical_data()` method:
        *   Fetches historical data from Binance using the `Client`.
        *   Converts the data to a Pandas DataFrame.
    *   `apply_strategy()` method:
        *   Applies the strategy to generate signals.

### Additional Considerations

*   **Strategy Pattern for Data Sources:** Abstract the data fetching into a `DataSourceStrategy` interface, allowing for different data sources (CSV files, other exchanges, etc.).
*   **Make BacktestHelper Stateless:** Implement `BacktestHelper` as a stateless utility class with static methods.
*   **Result Object Pattern:** Return a standardized `BacktestResult` object from `BacktestExecutor.run()` rather than a raw DataFrame.
*   **Event-Based Progress Updates:** Consider an event system for providing progress updates for long-running backtests (future enhancement).

## Implementation Plan

1.  Create the `BacktestConfiguration` class first.
2.  Implement the `BacktestHelper` with the shared functionality.
3.  Refactor `BacktestExecutor` to use the helper and configuration.
4.  Finally, update `Backtester` to orchestrate the new components.

## Class Diagram

```mermaid
classDiagram
    class Backtester {
        -config: BacktestConfiguration
        +Backtester(config: BacktestConfiguration)
        +run(): BacktestResult
    }
    class BacktestExecutor {
        -config: BacktestConfiguration
        +BacktestExecutor(config: BacktestConfiguration)
        +run(): BacktestResult
    }
    class BacktestHelper {
        +fetch_historical_data(config: BacktestConfiguration): DataFrame
        +apply_strategy(data: DataFrame, strategy: Strategy): DataFrame
    }
    class BacktestConfiguration {
        -client: Client
        -strategy: Strategy
        -start_date: str
        -end_date: str
        +BacktestConfiguration(client: Client, strategy: Strategy, start_date: str, end_date: str)
    }
    interface DataSourceStrategy {
        +get_data(symbol: str, interval: str, start_date: str, end_date: str): DataFrame
    }
    class BinanceDataSourceStrategy {
        +get_data(symbol: str, interval: str, start_date: str, end_date: str): DataFrame
    }
    class BacktestResult {
        -signals: DataFrame
        -metrics: Dict[str, Any]
        +BacktestResult(signals: DataFrame, metrics: Dict[str, Any])
    }
    Backtester --|> BacktestExecutor : uses
    BacktestExecutor --|> BacktestHelper : uses
    BacktestExecutor --|> BacktestConfiguration : uses
    BacktestExecutor --|> DataSourceStrategy : uses
    BinanceDataSourceStrategy --|> DataSourceStrategy : implements
    Backtester --|> BacktestResult : returns
    BacktestExecutor --|> BacktestResult : returns