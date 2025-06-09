import pytest
from PerformanceAnalyzer import PerformanceAnalyzer


@pytest.mark.asyncio
async def test_calculate_performance(monkeypatch, sample_performance_data):
    analyzer = PerformanceAnalyzer()

    async def dummy_store(metrics):
        dummy_store.saved = metrics

    monkeypatch.setattr(analyzer, "_store_metrics", dummy_store)
    results = await analyzer.calculate_performance(sample_performance_data)

    assert "metrics" in results
    assert results["metrics"]["final_capital"] >= analyzer.initial_capital
    assert hasattr(dummy_store, "saved")
