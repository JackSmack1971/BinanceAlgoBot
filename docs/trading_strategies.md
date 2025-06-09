# Trading Strategies

This project implements several example strategies. All strategies inherit from `BaseStrategy` and share a similar interface.

## BTCStrategy
Uses EMA, RSI, ATR and VWAP indicators to generate buy and sell signals.
The logic is summarised as:

1. Generate indicators for the current symbol.
2. Enter a long position when price crosses above VWAP and RSI exceeds the buy threshold.
3. Enter a short position when price crosses below VWAP and RSI is below the sell threshold.
4. Close positions based on ATR expansion or contraction conditions.

## EMACrossoverStrategy
A simple moving average crossover approach comparing fast and slow EMAs.

## MACDStrategy
Trades based on MACD histogram crossovers and trend confirmation.
