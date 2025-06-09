# Trading Bot Contributor Guide

## Architecture Overview
This is a high-frequency trading bot built with modern Python async patterns:
- `src/trading/strategies/` - Strategy implementations using Protocol pattern
- `src/execution/` - Smart order routing with market microstructure awareness
- `src/risk/` - Real-time risk management with circuit breakers
- `src/monitoring/` - Production observability with Prometheus/Grafana

## Development Standards
- **Type Safety**: All code must use comprehensive type hints (mypy --strict)
- **Async First**: Use async/await patterns, never blocking calls
- **Error Handling**: Implement circuit breakers and graceful degradation
- **Security**: Input validation with Pydantic, parameterized SQL queries
- **Performance**: Numba JIT for indicators, Redis caching for computations

## Testing Requirements
- Unit tests with 90%+ coverage using pytest and asyncio
- Integration tests for database and external API connections
- Performance benchmarks for all critical algorithms
- Security testing with malicious input validation

## Deployment Process
1. Run full test suite: `pytest tests/ --cov=src/ --cov-report=html`
2. Type checking: `mypy src/ --strict`
3. Security scan: `bandit -r src/`
4. Performance validation: `python scripts/benchmark.py`
5. Docker build and deploy: `./scripts/deploy.sh production`
