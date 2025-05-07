-- Table: market_data
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    close_time TIMESTAMP NOT NULL,
    quote_asset_volume DECIMAL(20, 8) NOT NULL,
    trades INTEGER NOT NULL,
    taker_buy_base DECIMAL(20, 8) NOT NULL,
    taker_buy_quote DECIMAL(20, 8) NOT NULL,
    ignored DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX market_data_symbol_interval_timestamp_idx ON market_data (symbol, interval, timestamp);

-- Implement partitioning by date ranges to improve query performance.
-- Implement an automated process to archive or delete market data beyond the 1-2 year retention period to control database size.

-- Table: indicators
CREATE TABLE indicators (
    id SERIAL PRIMARY KEY,
    market_data_id INTEGER REFERENCES market_data(id),
    ema DECIMAL(20, 8),
    rsi DECIMAL(20, 8),
    atr DECIMAL(20, 8),
    vwap DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: signals
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    market_data_id INTEGER REFERENCES market_data(id),
    strategy_id INTEGER REFERENCES strategies(id),
    signal INTEGER NOT NULL,  -- e.g., 1 for buy, -1 for sell, 0 for hold
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: trade_history
CREATE TABLE trade_history (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    position_type VARCHAR(10) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8) NOT NULL,
    profit_pct DECIMAL(20, 8) NOT NULL,
    duration DECIMAL(20, 8) NOT NULL,
    commission_fee DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX trade_history_strategy_id_entry_time_exit_time_idx ON trade_history (strategy_id, entry_time, exit_time);

-- Table: performance_metrics
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER REFERENCES trade_history(id),
    initial_capital DECIMAL(20, 8) NOT NULL,
    final_capital DECIMAL(20, 8) NOT NULL,
    total_return DECIMAL(20, 8) NOT NULL,
    annual_return DECIMAL(20, 8) NOT NULL,
    max_drawdown DECIMAL(20, 8) NOT NULL,
    sharpe_ratio DECIMAL(20, 8) NOT NULL,
    win_rate DECIMAL(20, 8) NOT NULL,
    avg_profit_pct DECIMAL(20, 8) NOT NULL,
    risk_reward_ratio DECIMAL(20, 8) NOT NULL,
    profit_factor DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: backtest_results
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    initial_capital DECIMAL(20, 8) NOT NULL,
    final_capital DECIMAL(20, 8) NOT NULL,
    total_return DECIMAL(20, 8) NOT NULL,
    annual_return DECIMAL(20, 8) NOT NULL,
    max_drawdown DECIMAL(20, 8) NOT NULL,
    sharpe_ratio DECIMAL(20, 8) NOT NULL,
    win_rate DECIMAL(20, 8) NOT NULL,
    avg_profit_pct DECIMAL(20, 8) NOT NULL,
    risk_reward_ratio DECIMAL(20, 8) NOT NULL,
    profit_factor DECIMAL(20, 8) NOT NULL,
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: configurations
CREATE TABLE configurations (
    id SERIAL PRIMARY KEY,
    config_name VARCHAR(255) NOT NULL,
    config_value TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX configurations_config_name_version_idx ON configurations (config_name, version);

-- Table: risk_parameters
CREATE TABLE risk_parameters (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    max_risk_per_trade DECIMAL(20, 8) NOT NULL,
    max_open_trades INTEGER NOT NULL,
    stop_loss_percentage DECIMAL(20, 8) NOT NULL,
    take_profit_percentage DECIMAL(20, 8) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: strategies
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(255) NOT NULL,
    strategy_type VARCHAR(255) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    initial_balance DECIMAL(20, 8) NOT NULL,
    risk_per_trade DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);