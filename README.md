# BinanceAlgoBot
# Algorithmic Trading Bot for Binance

## Project Description

This project is an algorithmic trading bot that operates on the Binance platform. It uses Python to interact with the Binance API and implements a trading strategy based on four indicators: EMA, RSI, ATR, and VWAP. The bot operates on a 15-minute timeframe for the BTCUSD pair.

## Table of Contents

1. [Installation](#installation)
2. [Setup](#setup)
3. [Configuration](#configuration)
4. [Usage](#usage)
3. [Contributing](#contributing)
4. [Tests](#tests)
5. [Credits](#credits)
6. [License](#license)

## Installation

To install and run the project, you need to have Python installed on your machine. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
pip install -r requirements.txt
```

### External Dependencies

The project relies on the following external Python libraries:

*   **numpy**: Used for numerical computations and array operations.
*   **pandas**: Used for data analysis and manipulation.
*   **python\_binance**: Used for interacting with the Binance API.
*   **ta**: Used for technical analysis indicators.
*   **psycopg2-binary**: Used for connecting to PostgreSQL database.
*   **pytest**: Used for testing.
*   **matplotlib**: Used for data visualization.
*   **backtester**: Used for backtesting trading strategies.

These dependencies are listed in the `requirements.txt` file.

## Setup

After installing the dependencies, copy the example environment file and provide your Binance credentials:

```bash
cp .env.example .env
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"
export DATABASE_URL="postgresql://user:password@host:5432/db"
```

This project never stores credentials in the repository. Always use environment variables.

## Configuration

Create a `.env` file based on `.env.example` and set your Binance API credentials before running the bot:

```bash
cp .env.example .env
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"
export DATABASE_URL="postgresql://user:password@host:5432/db"
```

The provided `config.json` keeps these values empty to avoid accidentally committing secrets.

Example `config.json`:

```json
{
    "api_key": "",
    "secret_key": "",
    "trade_symbol": "BTCUSDT",
    "data_interval": "1m"
}
```

## Usage

Run a backtest from the command line:

```bash
python main.py --mode backtest --symbol BTCUSDT --interval 15m --strategy btc
```

For live trading ensure your API keys are set and run:

```bash
python main.py --mode live --symbol BTCUSDT --interval 15m --strategy btc
```

