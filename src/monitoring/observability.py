from __future__ import annotations

import asyncio
import os

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import sentry_sdk


class MonitoringSetupError(Exception):
    """Raised when monitoring initialization fails."""


ORDERS_TOTAL = Counter("orders_total", "Total orders", ["strategy"])
SLIPPAGE_HIST = Histogram(
    "slippage_histogram", "Order slippage", ["strategy"], buckets=(0.0, 0.1, 0.2, 0.5, 1.0)
)
PORTFOLIO_VALUE = Gauge("portfolio_value_gauge", "Portfolio value", ["account"])
CPU_USAGE = Gauge("cpu_usage", "CPU usage percentage")
MEMORY_USAGE = Gauge("memory_usage", "Memory usage percentage")
ERROR_COUNTER = Counter("error_total", "Application errors", ["type"])
LATENCY_HIST = Histogram(
    "latency_distribution", "Request latency", buckets=(0.01, 0.1, 0.5, 1, 2)
)
PNL_GAUGE = Gauge("pnl", "Profit and loss", ["account"])
RISK_EXPOSURE = Gauge("risk_exposure", "Current risk exposure", ["account"])
MARKET_DATA_LATENCY = Histogram(
    "market_data_latency", "Market data latency", buckets=(0.01, 0.05, 0.1, 0.5)
)


async def start_metrics_server(port: int = 8080) -> None:
    """Start Prometheus metrics server."""
    try:
        await asyncio.to_thread(start_http_server, port)
    except Exception as exc:  # pragma: no cover - rarely triggered
        raise MonitoringSetupError("Prometheus failed") from exc


def setup_tracing(service: str) -> None:
    """Configure Jaeger tracing."""
    try:
        exporter = JaegerExporter(hostname=os.getenv("JAEGER_HOST", "localhost"))
        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: service})
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception as exc:  # pragma: no cover - rarely triggered
        raise MonitoringSetupError("Tracing setup failed") from exc


def setup_sentry() -> None:
    """Initialize Sentry if DSN provided."""
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return
    try:
        sentry_sdk.init(dsn=dsn, traces_sample_rate=1.0)
    except Exception as exc:  # pragma: no cover - rarely triggered
        raise MonitoringSetupError("Sentry setup failed") from exc


async def record_system_metrics() -> None:
    """Update CPU and memory metrics."""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
