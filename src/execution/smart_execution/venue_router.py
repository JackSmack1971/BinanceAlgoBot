from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Venue:
    name: str
    fee: float
    latency: float
    fill_rate: float


class VenueRouter:
    def __init__(self, venues: List[Venue]) -> None:
        self.venues = venues

    async def select_best_venue(self) -> Venue:
        await asyncio.sleep(0)
        scores: Dict[str, float] = {}
        for venue in self.venues:
            score = venue.fill_rate - venue.fee - venue.latency
            scores[venue.name] = score
        best = max(self.venues, key=lambda v: scores[v.name])
        return best
