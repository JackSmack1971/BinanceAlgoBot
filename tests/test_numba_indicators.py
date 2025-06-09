import numpy as np
import pandas as pd
import pytest

from src.indicators.numba_indicators import (
    NumbaIndicatorEngine,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_macd,
)


@pytest.fixture
def price_series() -> np.ndarray:
    np.random.seed(0)
    return np.random.rand(1000) * 100


@pytest.fixture
def ohlc(price_series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = price_series + np.random.rand(price_series.size)
    low = price_series - np.random.rand(price_series.size)
    return high, low, price_series


@pytest.mark.asyncio
async def test_sma_accuracy(price_series: np.ndarray):
    pandas_res = pd.Series(price_series).rolling(10).mean().to_numpy()
    engine = NumbaIndicatorEngine()
    numba_res = await engine.sma(price_series, 10)
    assert np.allclose(pandas_res, numba_res, equal_nan=True)


@pytest.mark.asyncio
async def test_ema_accuracy(price_series: np.ndarray):
    pandas_res = pd.Series(price_series).ewm(span=10, adjust=False).mean().to_numpy()
    engine = NumbaIndicatorEngine()
    numba_res = await engine.ema(price_series, 10)
    assert np.allclose(pandas_res, numba_res, atol=1e-5)


@pytest.mark.asyncio
async def test_rsi_accuracy(price_series: np.ndarray):
    diff = pd.Series(price_series).diff()
    gain = diff.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -diff.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    pandas_rsi = 100 - 100 / (1 + rs)
    engine = NumbaIndicatorEngine()
    numba_res = await engine.rsi(price_series, 14)
    assert np.allclose(pandas_rsi.to_numpy(), numba_res, equal_nan=True, atol=1e-3)


@pytest.mark.asyncio
async def test_atr_accuracy(ohlc: tuple[np.ndarray, np.ndarray, np.ndarray]):
    high, low, close = ohlc
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    tr = df['high'] - df['low']
    tr = np.maximum(tr, (df['high'] - df['close'].shift()).abs())
    tr = np.maximum(tr, (df['low'] - df['close'].shift()).abs())
    pandas_atr = tr.ewm(alpha=1/14, adjust=False).mean()
    engine = NumbaIndicatorEngine()
    numba_res = await engine.atr(high, low, close, 14)
    assert np.allclose(pandas_atr.to_numpy(), numba_res, equal_nan=True, atol=1e-5)


@pytest.mark.asyncio
async def test_macd_accuracy(price_series: np.ndarray):
    ema_fast = pd.Series(price_series).ewm(span=12, adjust=False).mean()
    ema_slow = pd.Series(price_series).ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    engine = NumbaIndicatorEngine()
    numba_macd, numba_signal = await engine.macd(price_series)
    assert np.allclose(macd_line.to_numpy(), numba_macd, atol=1e-5)
    assert np.allclose(signal_line.to_numpy(), numba_signal, atol=1e-5)


def test_numba_speed(price_series: np.ndarray):
    pandas_series = pd.Series(price_series)
    calculate_ema(price_series, 10)  # compile
    import time
    start = time.perf_counter()
    calculate_ema(price_series, 10)
    numba_time = time.perf_counter() - start
    start = time.perf_counter()
    pandas_series.ewm(span=10, adjust=False).mean()
    pandas_time = time.perf_counter() - start
    assert numba_time < pandas_time / 10
