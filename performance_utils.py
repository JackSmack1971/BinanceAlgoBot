import logging
import psutil

logger = logging.getLogger(__name__)


def log_memory_usage(prefix: str = "") -> None:
    """Log current process memory usage."""
    try:
        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024)
        logger.info("%sMemory usage: %.2f MB", prefix, mem)
    except Exception as exc:  # pragma: no cover - psutil errors
        logger.error("Memory usage check failed: %s", exc)

