import pytest

from src.execution.smart_execution.venue_router import Venue, VenueRouter


@pytest.mark.asyncio
async def test_select_best_venue():
    venues = [
        Venue("A", fee=0.001, latency=0.1, fill_rate=0.9),
        Venue("B", fee=0.002, latency=0.05, fill_rate=0.95),
    ]
    router = VenueRouter(venues)
    best = await router.select_best_venue()
    assert best.name == "B"
