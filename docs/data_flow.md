# Data Flow During a Backtest

The following describes the typical data flow during a backtest:

1.  **Historical Data Acquisition:** The `BacktestExecutor` retrieves historical market data from the Binance API.
2.  **Data Transformation:** The raw kline data is transformed into a structured Pandas DataFrame.
3.  **Technical Indicator Calculation:** The trading strategy processes this price data to calculate technical indicators.
4.  **Trading Signal Generation:** Each strategy implementation has specific trading rules that analyze the calculated indicators to generate buy, sell, or hold signals.
5.  **Position Tracking:** After generating signals, the strategy adds a 'position' column to track the current position over time based on signals.
6.  **Performance Analysis:** The `PerformanceAnalyzer` calculates returns based on the positions and price changes. It applies commission costs at position changes, calculates cumulative returns and portfolio value over time, and tracks drawdowns.
7.  **Reporting and Visualization:** The `VisualizationService` creates plots showing price data with indicators, buy/sell signals, portfolio value, drawdowns, and strategy performance compared to a buy-and-hold approach. The `ReportGenerator` creates detailed reports with performance metrics, trade statistics, monthly returns, and comparison to benchmark strategies.