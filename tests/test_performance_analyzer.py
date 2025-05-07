import pytest
from PerformanceAnalyzer import PerformanceAnalyzer
from unittest.mock import MagicMock

def test_performance_analyzer_creation():
    analyzer = PerformanceAnalyzer()
    assert analyzer is not None

def test_performance_analyzer_calculate_metrics():
    analyzer = PerformanceAnalyzer()
    signals = MagicMock()
    metrics = analyzer.calculate_performance(signals)
    assert metrics is not None