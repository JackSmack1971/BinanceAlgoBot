from __future__ import annotations

import logging
from typing import Iterable

import numpy as np


class RiskCalculator:
    def __init__(self, confidence: float = 0.95) -> None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence
        self.logger = logging.getLogger(__name__)
        fmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def value_at_risk(self, returns: Iterable[float]) -> float:
        arr = np.asarray(list(returns), dtype=float)
        if arr.size == 0:
            raise ValueError("returns cannot be empty")
        var = -np.percentile(arr, (1 - self.confidence) * 100)
        self.logger.info("Computed VaR: %.6f", var)
        return float(var)

    def max_drawdown(self, equity_curve: Iterable[float]) -> float:
        arr = np.asarray(list(equity_curve), dtype=float)
        if arr.size == 0:
            raise ValueError("equity_curve cannot be empty")
        peak = np.maximum.accumulate(arr)
        drawdown = (arr - peak) / peak
        mdd = drawdown.min()
        self.logger.info("Computed Max Drawdown: %.6f", mdd)
        return float(mdd)

    def position_concentration(self, positions: Iterable[float]) -> float:
        arr = np.abs(np.asarray(list(positions), dtype=float))
        if arr.sum() == 0:
            return 0.0
        concentration = arr.max() / arr.sum()
        self.logger.info("Computed Position Concentration: %.6f", concentration)
        return float(concentration)
