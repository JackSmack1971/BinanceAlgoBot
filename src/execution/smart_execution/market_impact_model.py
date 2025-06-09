from __future__ import annotations

from dataclasses import dataclass

@dataclass
class MarketImpactModel:
    base_impact: float = 0.0001

    def estimate(self, quantity: float) -> float:
        """Estimate market impact cost."""
        return self.base_impact * quantity
