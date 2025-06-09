"""Monitoring utilities with lazy imports to avoid heavy dependencies."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = [
    "start_metrics_server",
    "setup_tracing",
    "setup_sentry",
    "record_system_metrics",
    "HealthServer",
    "PerformanceTracker",
    "AuditTrailRecorder",
]

_module_map = {
    "start_metrics_server": (".observability", "start_metrics_server"),
    "setup_tracing": (".observability", "setup_tracing"),
    "setup_sentry": (".observability", "setup_sentry"),
    "record_system_metrics": (".observability", "record_system_metrics"),
    "HealthServer": (".health_checks", "HealthServer"),
    "PerformanceTracker": (".performance_tracking", "PerformanceTracker"),
    "AuditTrailRecorder": (".audit_trail", "AuditTrailRecorder"),
}


def __getattr__(name: str) -> Any:
    if name not in _module_map:
        raise AttributeError(name)
    module_path, attr = _module_map[name]
    module: ModuleType = import_module(module_path, __name__)
    return getattr(module, attr)
