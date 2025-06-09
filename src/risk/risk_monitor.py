from __future__ import annotations

import asyncio
from typing import List

import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest


class RiskMonitorError(Exception):
    """Raised when risk monitoring fails."""


class RiskEvent(BaseModel):
    position_value: float = Field(..., ge=0)
    trade_volume: float = Field(..., ge=0)
    pnl: float


class RiskMonitor:
    def __init__(self, window_size: int = 100) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._data: List[list[float]] = []
        self._model = IsolationForest(contamination=0.05, random_state=42)
        self._lock = asyncio.Lock()

    async def record_event(self, event: RiskEvent) -> None:
        async with self._lock:
            self._data.append([event.position_value, event.trade_volume, event.pnl])
            if len(self._data) > self.window_size:
                self._data.pop(0)

    async def detect_anomaly(self) -> bool:
        async with self._lock:
            if len(self._data) < 10:
                return False
            data = np.asarray(self._data, dtype=float)
            try:
                self._model.fit(data)
                result = self._model.predict(data[-1:].reshape(1, -1))
                return bool(result[0] == -1)
            except Exception as exc:
                raise RiskMonitorError("anomaly detection failed") from exc
