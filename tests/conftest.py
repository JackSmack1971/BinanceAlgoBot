import os
import sys
from typing import Generator

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_strategy_data() -> pd.DataFrame:
    """Simple market data for strategy tests."""
    return pd.DataFrame(
        {
            "close": [100, 105, 102],
            "vwap": [99, 100, 101],
            "rsi": [40, 60, 55],
            "ema": [100, 101, 102],
            "atr": [1, 1, 1],
        }
    )


@pytest.fixture
def sample_performance_data() -> pd.DataFrame:
    """Minimal performance data for analyzer tests."""
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "close": [100, 110, 105],
            "signal": [0, 1, -1],
            "position": [0, 1, 0],
        },
        index=index,
    )
