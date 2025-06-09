# Database Schema

## Tables

### `strategies`

| Column      | Data Type     | Description                               |
| ----------- | ----------- | ----------------------------------------- |
| `id`        | `INTEGER`   | Primary key                               |
| `name`      | `VARCHAR(255)`| Name of the strategy                      |
| `description` | `TEXT`      | Description of the strategy               |
| `created_at`| `TIMESTAMP` | Timestamp of when the strategy was created |

### `risk_parameters`

| Column              | Data Type     | Description                                                                 |
| ------------------- | ----------- | --------------------------------------------------------------------------- |
| `id`                | `INTEGER`   | Primary key                                                                 |
| `strategy_id`       | `INTEGER`   | Foreign key referencing `strategies.id`                                     |
| `max_risk_per_trade`| `DECIMAL(20, 8)`| Maximum percentage of account balance risked per trade                      |
| `max_open_trades`   | `INTEGER`   | Maximum number of concurrent positions allowed                               |
| `stop_loss_percentage`| `DECIMAL(20, 8)`| Maximum allowable loss per trade                                            |
| `take_profit_percentage`| `DECIMAL(20, 8)`| Target profit level per trade                                               |
| `version`           | `INTEGER`   | Version of the risk parameters                                              |
| `created_at`        | `TIMESTAMP` | Timestamp of when the risk parameters were created                           |

### `market_data`

| Column      | Data Type     | Description                     |
| ----------- | ----------- | ------------------------------- |
| `id`        | `INTEGER`   | Primary key                     |
| `symbol`    | `VARCHAR(255)`| Trading symbol                  |
| `interval`  | `VARCHAR(255)`| Timeframe interval              |
| `timestamp` | `TIMESTAMP` | Timestamp of the market data    |
| `open`      | `DECIMAL(20, 8)`| Open price                      |
| `high`      | `DECIMAL(20, 8)`| High price                      |
| `low`       | `DECIMAL(20, 8)`| Low price                       |
| `close`     | `DECIMAL(20, 8)`| Close price                     |
| `volume`    | `DECIMAL(20, 8)`| Volume                          |

### `backtest_results`

| Column          | Data Type     | Description                               |
| --------------- | ----------- | ----------------------------------------- |
| `id`            | `INTEGER`   | Primary key                               |
| `strategy_id`   | `INTEGER`   | Foreign key referencing `strategies.id`   |
| `start_date`    | `DATE`      | Start date of the backtest                |
| `end_date`      | `DATE`      | End date of the backtest                  |
| `initial_capital` | `DECIMAL(20, 8)`| Initial capital for the backtest          |
| `final_capital`   | `DECIMAL(20, 8)`| Final capital after the backtest          |
| `total_return`    | `DECIMAL(20, 8)`| Total return of the backtest              |
| `max_drawdown`    | `DECIMAL(20, 8)`| Maximum drawdown during the backtest      |
| `sharpe_ratio`    | `DECIMAL(20, 8)`| Sharpe ratio of the backtest              |
| `created_at`    | `TIMESTAMP` | Timestamp of when the result was created |

