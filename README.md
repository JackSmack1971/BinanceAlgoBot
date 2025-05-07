# BinanceAlgoBot
# Algorithmic Trading Bot for Binance

## Project Description

This project is an algorithmic trading bot that operates on the Binance platform. It uses Python to interact with the Binance API and implements a trading strategy based on four indicators: EMA, RSI, ATR, and VWAP. The bot operates on a 15-minute timeframe for the BTCUSD pair.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
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
