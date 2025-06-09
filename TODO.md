# Enhanced Trading Bot Optimization Guide for Production-Grade Systems (2025)

## 🚨 Critical Priority Fixes (Week 1)

### Prompt 1: Implement Comprehensive Input Validation Security Layer
```
There are critical security vulnerabilities in the trading bot codebase related to input validation and SQL injection prevention. 

Please examine these files and implement comprehensive input validation with modern Python patterns:
- `validation.py` - enhance existing validation functions using Pydantic v2
- `database/market_data_repository.py` - add parameterized query validation with asyncpg
- `TradingOrchestrator.py` - add input sanitization for trading parameters

Create a new `TradingInputValidator` class using Pydantic v2 with these methods:

```python
from pydantic import BaseModel, Field, validator, root_validator
from decimal import Decimal
from typing import Literal, Optional
from datetime import datetime
import re

class TradingSymbolValidator(BaseModel):
    symbol: str = Field(..., regex=r'^[A-Z]{2,10}USD?T?$', description="Valid trading symbol")
    
    @validator('symbol')
    def validate_symbol_whitelist(cls, v):
        # Implement whitelist validation against approved symbols
        approved_symbols = {"BTCUSDT", "ETHUSDT", "ADAUSDT"}  # Load from config
        if v not in approved_symbols:
            raise ValueError(f"Symbol {v} not in approved list")
        return v

class TradingOrderValidator(BaseModel):
    symbol: str = Field(..., regex=r'^[A-Z]{2,10}USD?T?$')
    side: Literal['BUY', 'SELL']
    quantity: Decimal = Field(..., gt=0, decimal_places=8)
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    order_type: Literal['MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT']
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('quantity')
    def validate_quantity_bounds(cls, v):
        if v > Decimal('1000000'):  # Max position size
            raise ValueError("Quantity exceeds maximum allowed")
        return v
    
    @root_validator
    def validate_order_consistency(cls, values):
        if values.get('order_type') == 'LIMIT' and not values.get('price'):
            raise ValueError("Limit orders require price")
        return values

# Enhanced database operations with asyncpg parameterized queries
class SecureMarketDataRepository:
    async def insert_trade_data(self, trade_data: TradingOrderValidator):
        query = """
        INSERT INTO trades (symbol, side, quantity, price, order_type, timestamp)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                query, 
                trade_data.symbol, 
                trade_data.side, 
                trade_data.quantity, 
                trade_data.price, 
                trade_data.order_type, 
                trade_data.timestamp
            )
```

Test the implementation with:
```python
# Test malicious inputs - all should raise ValidationError
try:
    TradingSymbolValidator(symbol="'; DROP TABLE trades; --")
except ValidationError as e:
    print("SQL injection attempt blocked")

try:
    TradingOrderValidator(quantity=Decimal('-100.50'))
except ValidationError as e:
    print("Negative quantity blocked")
```

Ensure all validation failures raise appropriate `ValidationError` exceptions with descriptive messages.
```

### Prompt 2: Modernize Async Implementation with 2025 Best Practices
```
The codebase has several async anti-patterns. Implement modern asyncio patterns based on 2025 best practices:

In `data_feed.py`:
- Replace blocking calls with proper async patterns using `asyncio.to_thread()`
- Implement connection pooling with asyncpg for database operations
- Add proper timeout handling with `asyncio.wait_for()`

```python
import asyncio
import asyncpg
from contextlib import asynccontextmanager
from typing import AsyncGenerator

class ModernDataFeed:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
    
    async def get_market_data(self, symbol: str) -> dict:
        """Get market data with proper async patterns."""
        async with self.semaphore:
            try:
                # Use asyncio.wait_for for timeout handling
                data = await asyncio.wait_for(
                    self._fetch_external_data(symbol),
                    timeout=5.0
                )
                return data
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching data for {symbol}")
                raise
    
    async def _fetch_external_data(self, symbol: str) -> dict:
        """Fetch data from external API with proper async client."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.exchange.com/v1/ticker/{symbol}") as response:
                return await response.json()
    
    async def process_indicators_concurrent(self, price_data: list) -> dict:
        """Process multiple indicators concurrently."""
        tasks = [
            asyncio.create_task(self.calculate_ema(price_data, 20)),
            asyncio.create_task(self.calculate_rsi(price_data, 14)),
            asyncio.create_task(self.calculate_atr(price_data, 14))
        ]
        
        # Use gather with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        indicators = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Indicator calculation failed: {result}")
                indicators[f"indicator_{i}"] = None
            else:
                indicators[f"indicator_{i}"] = result
        
        return indicators

# Database connection with modern asyncpg patterns
class AsyncDatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize_pool(self):
        """Initialize connection pool with optimized settings."""
        self.pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=20,
            max_size=100,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=30,
            server_settings={
                'jit': 'off',  # Disable JIT for consistent performance
                'application_name': 'trading_bot'
            }
        )
    
    @asynccontextmanager
    async def acquire_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Context manager for safe connection handling."""
        if not self.pool:
            raise RuntimeError("Pool not initialized")
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
            except Exception:
                # Connection will be automatically returned to pool
                raise
    
    async def close_pool(self):
        """Gracefully close the connection pool."""
        if self.pool:
            await self.pool.close()
```

Update all blocking operations to use proper async patterns:
```python
# OLD - Blocking pattern
def process_data(self):
    result = self.client.get_klines(symbol, interval, limit)
    return result

# NEW - Modern async pattern
async def process_data(self):
    async with self.rate_limiter:
        try:
            result = await asyncio.wait_for(
                self.async_client.get_klines(symbol, interval, limit),
                timeout=10.0
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Data fetch timeout")
            raise
```

Test all changes with:
```bash
python -m pytest tests/ -v --asyncio-mode=auto
```
```

### Prompt 3: Implement Advanced Trading Circuit Breaker System
```
Implement a sophisticated circuit breaker and kill switch system using modern Python patterns.

Create `risk/circuit_breaker.py`:

```python
import asyncio
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # Successes needed in half-open state
    timeout: float = 10.0

@dataclass
class CircuitBreakerMetrics:
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    state_changes: list = field(default_factory=list)

class TradingCircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.metrics = CircuitBreakerMetrics()
        self.state = CircuitState.CLOSED
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
            
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
                await self._on_success()
                return result
                
            except Exception as e:
                await self._on_failure(e)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.metrics.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.metrics.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    async def _transition_to_half_open(self):
        """Transition to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.metrics.success_count = 0
        self._log_state_change("HALF_OPEN", "Attempting recovery")
    
    async def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.metrics.success_count += 1
            if self.metrics.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.metrics.failure_count = 0
                self._log_state_change("CLOSED", "Circuit breaker reset")
        elif self.state == CircuitState.CLOSED:
            self.metrics.failure_count = 0
    
    async def _on_failure(self, error: Exception):
        """Handle failed execution."""
        self.metrics.failure_count += 1
        self.metrics.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.metrics.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._log_state_change("OPEN", f"Failure threshold reached: {error}")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._log_state_change("OPEN", f"Half-open test failed: {error}")
    
    def _log_state_change(self, new_state: str, reason: str):
        """Log state changes with metrics."""
        timestamp = datetime.utcnow()
        self.metrics.state_changes.append({
            'timestamp': timestamp,
            'state': new_state,
            'reason': reason,
            'failure_count': self.metrics.failure_count
        })
        self.logger.warning(f"Circuit breaker {new_state}: {reason}")

class TradingKillSwitch:
    def __init__(self, execution_engine, position_manager):
        self.execution_engine = execution_engine
        self.position_manager = position_manager
        self.is_active = False
        self.activation_time: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
    
    async def activate(self, reason: str):
        """Activate emergency kill switch."""
        if self.is_active:
            return
        
        self.is_active = True
        self.activation_time = datetime.utcnow()
        
        self.logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Cancel all pending orders
        await self._cancel_all_orders()
        
        # Close risky positions
        await self._close_risky_positions()
        
        # Send administrator alerts
        await self._send_admin_alerts(reason)
    
    async def _cancel_all_orders(self):
        """Cancel all pending orders immediately."""
        try:
            await self.execution_engine.cancel_all_orders()
            self.logger.info("All pending orders cancelled")
        except Exception as e:
            self.logger.error(f"Failed to cancel orders: {e}")
    
    async def _close_risky_positions(self):
        """Close positions based on risk metrics."""
        try:
            positions = await self.position_manager.get_open_positions()
            for position in positions:
                if position.unrealized_pnl_percent < -5.0:  # Close if losing > 5%
                    await self.position_manager.close_position(position.symbol)
        except Exception as e:
            self.logger.error(f"Failed to close positions: {e}")
    
    async def _send_admin_alerts(self, reason: str):
        """Send emergency notifications to administrators."""
        # Implement your alerting mechanism (email, Slack, etc.)
        pass

# Integration with TradingOrchestrator
class EnhancedTradingOrchestrator:
    def __init__(self):
        self.circuit_breaker = TradingCircuitBreaker(CircuitBreakerConfig())
        self.kill_switch = TradingKillSwitch(execution_engine, position_manager)
    
    async def execute_trading_operation(self, operation_func, *args, **kwargs):
        """Execute trading operation with circuit breaker protection."""
        if self.kill_switch.is_active:
            raise TradingSystemDisabledError("Kill switch is active")
        
        return await self.circuit_breaker.call(operation_func, *args, **kwargs)

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is in open state."""
    pass

class TradingSystemDisabledError(Exception):
    """Raised when trading system is disabled by kill switch."""
    pass
```

Test with simulated failures:
```python
async def test_circuit_breaker():
    async def failing_operation():
        raise Exception("Simulated failure")
    
    breaker = TradingCircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
    
    # Should fail and open circuit after 2 failures
    for _ in range(3):
        try:
            await breaker.call(failing_operation)
        except:
            pass
    
    assert breaker.state == CircuitState.OPEN
```
```

## 🔧 High Priority Performance Fixes (Week 2)

### Prompt 4: Optimize Database Performance with Modern Connection Pooling
```
Enhance database performance using asyncpg with optimized connection pooling for high-frequency trading.

Update `database/database_connection.py`:

```python
import asyncpg
import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import time

class OptimizedDatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        self._pool_stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'queries_executed': 0,
            'avg_query_time': 0.0
        }
    
    async def initialize_pool(self):
        """Initialize optimized connection pool."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            # Optimized pool settings for high-frequency trading
            min_size=20,                    # Always keep 20 connections ready
            max_size=100,                   # Scale up to 100 under load
            max_queries=50000,              # Queries per connection before replacement
            max_inactive_connection_lifetime=300,  # 5 minutes idle timeout
            command_timeout=30,             # 30 second query timeout
            server_settings={
                'jit': 'off',               # Disable JIT for consistent performance
                'application_name': 'trading_bot_hft',
                'tcp_keepalives_idle': '600',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3'
            },
            # Connection initialization
            init=self._init_connection
        )
        self.logger.info("Database pool initialized with optimized settings")
    
    async def _init_connection(self, connection: asyncpg.Connection):
        """Initialize each connection with optimizations."""
        # Set up prepared statements for common queries
        await connection.execute("SET synchronous_commit = off")  # For better performance
        await connection.execute("SET wal_buffers = '16MB'")
        
        # Prepare frequently used statements
        await connection.prepare("""
            INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """)
        
        self._pool_stats['connections_created'] += 1
    
    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire connection with automatic stats tracking."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        start_time = time.perf_counter()
        async with self.pool.acquire() as connection:
            try:
                yield connection
                # Track successful query
                query_time = time.perf_counter() - start_time
                self._update_query_stats(query_time)
            except Exception:
                self.logger.error("Database operation failed")
                raise
    
    def _update_query_stats(self, query_time: float):
        """Update query performance statistics."""
        self._pool_stats['queries_executed'] += 1
        # Calculate running average
        count = self._pool_stats['queries_executed']
        current_avg = self._pool_stats['avg_query_time']
        self._pool_stats['avg_query_time'] = (
            (current_avg * (count - 1) + query_time) / count
        )
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get detailed pool status for monitoring."""
        if not self.pool:
            return {'status': 'not_initialized'}
        
        return {
            'size': self.pool.get_size(),
            'free_connections': self.pool.get_size() - self.pool.get_busy_count(),
            'busy_connections': self.pool.get_busy_count(),
            'stats': self._pool_stats.copy()
        }

# Enhanced batch operations for market data
class HighPerformanceMarketDataRepository:
    def __init__(self, db_manager: OptimizedDatabaseManager):
        self.db_manager = db_manager
    
    async def batch_insert_market_data(self, market_data_batch: list) -> str:
        """High-performance batch insert using COPY."""
        async with self.db_manager.acquire_connection() as conn:
            # Use COPY for maximum throughput
            result = await conn.copy_records_to_table(
                'market_data',
                records=market_data_batch,
                columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            return result
    
    async def insert_trades_batch(self, trades: list) -> None:
        """Batch insert trades with transaction safety."""
        async with self.db_manager.acquire_connection() as conn:
            async with conn.transaction():
                # Use executemany for parameterized batch inserts
                await conn.executemany("""
                    INSERT INTO trades (symbol, side, quantity, price, timestamp)
                    VALUES ($1, $2, $3, $4, $5)
                """, trades)

# Create optimized indexes for time-series queries
OPTIMIZATION_INDEXES = [
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_time 
    ON market_data (symbol, timestamp DESC)
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_timestamp_brin 
    ON trades USING BRIN (timestamp) WITH (pages_per_range=128)
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_symbol_status
    ON positions (symbol, status) WHERE status = 'OPEN'
    """
]

async def apply_database_optimizations(db_manager: OptimizedDatabaseManager):
    """Apply database optimizations and indexes."""
    async with db_manager.acquire_connection() as conn:
        for index_sql in OPTIMIZATION_INDEXES:
            try:
                await conn.execute(index_sql)
                logging.info(f"Applied optimization: {index_sql.split()[0:6]}")
            except Exception as e:
                logging.warning(f"Failed to apply optimization: {e}")
```

Add performance monitoring:
```python
# Performance monitoring for database operations
class DatabasePerformanceMonitor:
    def __init__(self, db_manager: OptimizedDatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def monitor_pool_health(self):
        """Continuous monitoring of pool health."""
        while True:
            try:
                status = await self.db_manager.get_pool_status()
                
                # Alert if pool utilization is high
                if status.get('busy_connections', 0) / status.get('size', 1) > 0.8:
                    self.logger.warning("High database pool utilization detected")
                
                # Alert if average query time is high
                if status.get('stats', {}).get('avg_query_time', 0) > 1.0:
                    self.logger.warning("High database query latency detected")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Pool monitoring error: {e}")
                await asyncio.sleep(60)  # Back off on errors
```

Test with load simulation:
```python
async def stress_test_database():
    """Load test database operations."""
    db_manager = OptimizedDatabaseManager(DATABASE_URL)
    await db_manager.initialize_pool()
    
    # Simulate 10,000 concurrent market data inserts
    tasks = []
    for i in range(10000):
        task = asyncio.create_task(
            insert_market_data_point(db_manager, f"BTC{i%100}", time.time())
        )
        tasks.append(task)
    
    start_time = time.perf_counter()
    await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    
    print(f"Processed 10,000 inserts in {duration:.2f}s ({10000/duration:.0f} ops/sec)")
```
```

### Prompt 5: Implement Intelligent Redis Caching with Modern Patterns
```
Add intelligent caching for technical indicators using Redis with modern async patterns.

Create `cache/redis_cache_manager.py`:

```python
import asyncio
import redis.asyncio as redis
import json
import hashlib
import numpy as np
import zlib
from typing import Optional, Any, Dict, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

@dataclass
class CacheConfig:
    redis_url: str = "redis://localhost:6379"
    max_connections: int = 50
    default_ttl: int = 300  # 5 minutes
    compression_threshold: int = 1024  # Compress data > 1KB
    key_prefix: str = "trading_bot"

class IntelligentRedisCache:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.logger = logging.getLogger(__name__)
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize Redis connection pool with optimized settings."""
        self.redis_pool = redis.ConnectionPool.from_url(
            self.config.redis_url,
            max_connections=self.config.max_connections,
            retry_on_timeout=True,
            retry_on_error=[redis.BusyLoadingError, redis.ConnectionError],
            health_check_interval=30
        )
        
        self.redis_client = redis.Redis(
            connection_pool=self.redis_pool,
            decode_responses=False  # We handle encoding ourselves for better control
        )
        
        # Test connection
        await self.redis_client.ping()
        self.logger.info("Redis cache initialized successfully")
    
    def _generate_cache_key(self, symbol: str, indicator_type: str, **params) -> str:
        """Generate deterministic cache key with parameter hashing."""
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"{self.config.key_prefix}:{indicator_type}:{symbol}:{param_hash}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression."""
        if isinstance(data, np.ndarray):
            # Special handling for numpy arrays
            serialized = {
                'type': 'numpy',
                'data': data.tobytes(),
                'dtype': str(data.dtype),
                'shape': data.shape
            }
        else:
            serialized = {'type': 'json', 'data': data}
        
        json_data = json.dumps(serialized).encode('utf-8')
        
        # Compress if data is large
        if len(json_data) > self.config.compression_threshold:
            return zlib.compress(json_data)
        
        return json_data
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with decompression support."""
        try:
            # Try to decompress first
            try:
                decompressed = zlib.decompress(data)
            except zlib.error:
                decompressed = data
            
            deserialized = json.loads(decompressed.decode('utf-8'))
            
            if deserialized['type'] == 'numpy':
                return np.frombuffer(
                    deserialized['data'].encode('latin1'), 
                    dtype=deserialized['dtype']
                ).reshape(deserialized['shape'])
            else:
                return deserialized['data']
                
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            return None
    
    async def get(self, symbol: str, indicator_type: str, **params) -> Optional[Any]:
        """Get cached indicator data."""
        if not self.redis_client:
            return None
        
        cache_key = self._generate_cache_key(symbol, indicator_type, **params)
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                self._cache_stats['hits'] += 1
                return self._deserialize_data(cached_data)
            else:
                self._cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            self._cache_stats['errors'] += 1
            return None
    
    async def set(self, symbol: str, indicator_type: str, value: Any, 
                  ttl: Optional[int] = None, **params) -> bool:
        """Set cached indicator data."""
        if not self.redis_client:
            return False
        
        cache_key = self._generate_cache_key(symbol, indicator_type, **params)
        ttl = ttl or self.config.default_ttl
        
        try:
            serialized_data = self._serialize_data(value)
            await self.redis_client.setex(cache_key, ttl, serialized_data)
            self._cache_stats['sets'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            self._cache_stats['errors'] += 1
            return False
    
    async def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cached data for a symbol."""
        if not self.redis_client:
            return 0
        
        pattern = f"{self.config.key_prefix}:*:{symbol}:*"
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            self.logger.error(f"Cache invalidation error: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_operations = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_ratio = self._cache_stats['hits'] / total_operations if total_operations > 0 else 0
        
        redis_info = {}
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info()
            except Exception:
                pass
        
        return {
            'hit_ratio': hit_ratio,
            'stats': self._cache_stats.copy(),
            'redis_info': {
                'connected_clients': redis_info.get('connected_clients', 0),
                'used_memory_human': redis_info.get('used_memory_human', 'unknown'),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0)
            }
        }

# Enhanced indicators with intelligent caching
class CachedTechnicalIndicators:
    def __init__(self, cache: IntelligentRedisCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    async def get_ema(self, symbol: str, prices: np.ndarray, window: int) -> np.ndarray:
        """Get EMA with intelligent caching."""
        # Create data checksum for cache validation
        data_checksum = hashlib.md5(prices.tobytes()).hexdigest()[:8]
        
        cached_result = await self.cache.get(
            symbol, 'ema', 
            window=window, 
            checksum=data_checksum
        )
        
        if cached_result is not None:
            return cached_result
        
        # Calculate EMA
        ema = self._calculate_ema(prices, window)
        
        # Cache with appropriate TTL based on data freshness
        ttl = 300 if len(prices) > 1000 else 60  # Longer TTL for more data
        await self.cache.set(
            symbol, 'ema', ema, ttl,
            window=window, 
            checksum=data_checksum
        )
        
        return ema
    
    async def get_rsi(self, symbol: str, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Get RSI with caching."""
        data_checksum = hashlib.md5(prices.tobytes()).hexdigest()[:8]
        
        cached_result = await self.cache.get(
            symbol, 'rsi',
            window=window,
            checksum=data_checksum
        )
        
        if cached_result is not None:
            return cached_result
        
        rsi = self._calculate_rsi(prices, window)
        
        await self.cache.set(
            symbol, 'rsi', rsi, 300,
            window=window,
            checksum=data_checksum
        )
        
        return rsi
    
    def _calculate_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate EMA using efficient numpy operations."""
        alpha = 2.0 / (window + 1.0)
        ema = np.empty_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate RSI using efficient numpy operations."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(window)/window, mode='valid')
        avg_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
        
        rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with NaN for the first window-1 values
        return np.concatenate([np.full(window, np.nan), rsi])

# Cache warming strategy
class CacheWarmingManager:
    def __init__(self, cache: IntelligentRedisCache, indicators: CachedTechnicalIndicators):
        self.cache = cache
        self.indicators = indicators
        self.logger = logging.getLogger(__name__)
    
    async def warm_cache_for_symbols(self, symbols: list, price_data: Dict[str, np.ndarray]):
        """Pre-populate cache with commonly used indicators."""
        warming_tasks = []
        
        for symbol in symbols:
            if symbol not in price_data:
                continue
            
            prices = price_data[symbol]
            
            # Common indicator combinations
            tasks = [
                self.indicators.get_ema(symbol, prices, 9),
                self.indicators.get_ema(symbol, prices, 21),
                self.indicators.get_ema(symbol, prices, 50),
                self.indicators.get_rsi(symbol, prices, 14),
                self.indicators.get_rsi(symbol, prices, 21)
            ]
            
            warming_tasks.extend(tasks)
        
        try:
            await asyncio.gather(*warming_tasks, return_exceptions=True)
            self.logger.info(f"Cache warmed for {len(symbols)} symbols")
        except Exception as e:
            self.logger.error(f"Cache warming error: {e}")
```

Add cache monitoring:
```python
# Cache performance monitoring
async def monitor_cache_performance(cache: IntelligentRedisCache):
    """Monitor cache performance and alert on issues."""
    while True:
        try:
            stats = await cache.get_cache_stats()
            
            # Alert if hit ratio is low
            if stats['hit_ratio'] < 0.7:
                logging.warning(f"Low cache hit ratio: {stats['hit_ratio']:.2%}")
            
            # Alert if error rate is high
            error_rate = stats['stats']['errors'] / max(1, sum(stats['stats'].values()))
            if error_rate > 0.05:
                logging.warning(f"High cache error rate: {error_rate:.2%}")
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logging.error(f"Cache monitoring error: {e}")
            await asyncio.sleep(120)  # Back off on errors
```

Test cache functionality:
```python
async def test_cache_performance():
    """Test cache performance with realistic data."""
    config = CacheConfig(redis_url="redis://localhost:6379")
    cache = IntelligentRedisCache(config)
    await cache.initialize()
    
    indicators = CachedTechnicalIndicators(cache)
    
    # Generate test data
    prices = np.random.randn(1000).cumsum() + 100
    
    # Test cache miss (first call)
    start_time = time.perf_counter()
    ema1 = await indicators.get_ema("BTCUSDT", prices, 20)
    miss_time = time.perf_counter() - start_time
    
    # Test cache hit (second call)
    start_time = time.perf_counter()
    ema2 = await indicators.get_ema("BTCUSDT", prices, 20)
    hit_time = time.perf_counter() - start_time
    
    print(f"Cache miss time: {miss_time:.4f}s")
    print(f"Cache hit time: {hit_time:.4f}s")
    print(f"Speedup: {miss_time/hit_time:.1f}x")
    
    # Verify results are identical
    assert np.array_equal(ema1, ema2, equal_nan=True)
    
    # Check cache stats
    stats = await cache.get_cache_stats()
    assert stats['hit_ratio'] > 0.5
```
```

### Prompt 6: Optimize Memory Usage with Advanced DataFrame Techniques
```
Implement advanced memory optimization using modern pandas and numpy techniques.

Create `utils/memory_optimizer.py`:

```python
import pandas as pd
import numpy as np
import psutil
import gc
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import warnings

@dataclass
class MemoryUsageReport:
    before_mb: float
    after_mb: float
    reduction_mb: float
    reduction_percent: float
    dtypes_optimized: Dict[str, str]

class AdvancedMemoryOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._memory_threshold_mb = 1024  # Alert if DataFrame > 1GB
    
    def optimize_trading_dataframe(self, df: pd.DataFrame, 
                                   aggressive: bool = False) -> MemoryUsageReport:
        """Optimize DataFrame memory usage with advanced techniques."""
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        initial_dtypes = df.dtypes.to_dict()
        
        # Create copy to avoid modifying original
        optimized_df = df.copy()
        
        # 1. Optimize numeric columns
        optimized_df = self._optimize_numeric_columns(optimized_df)
        
        # 2. Optimize string columns with category conversion
        optimized_df = self._optimize_string_columns(optimized_df, aggressive)
        
        # 3. Optimize datetime columns
        optimized_df = self._optimize_datetime_columns(optimized_df)
        
        # 4. Remove unnecessary precision
        if aggressive:
            optimized_df = self._reduce_precision(optimized_df)
        
        final_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
        reduction_mb = initial_memory - final_memory
        reduction_percent = (reduction_mb / initial_memory) * 100 if initial_memory > 0 else 0
        
        final_dtypes = {col: str(dtype) for col, dtype in optimized_df.dtypes.items()}
        changed_dtypes = {
            col: f"{initial_dtypes[col]} -> {final_dtypes[col]}"
            for col in initial_dtypes
            if str(initial_dtypes[col]) != final_dtypes[col]
        }
        
        self.logger.info(f"Memory optimization: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
                        f"({reduction_percent:.1f}% reduction)")
        
        return MemoryUsageReport(
            before_mb=initial_memory,
            after_mb=final_memory,
            reduction_mb=reduction_mb,
            reduction_percent=reduction_percent,
            dtypes_optimized=changed_dtypes
        )
    
    def _optimize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize numeric columns by downcasting."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = df[col]
            
            # Skip if column has NaN values and is critical
            if col_data.isna().any() and col in ['price', 'quantity', 'value']:
                continue
            
            # Downcast integers
            if pd.api.types.is_integer_dtype(col_data):
                df[col] = pd.to_numeric(col_data, downcast='integer')
            
            # Downcast floats
            elif pd.api.types.is_float_dtype(col_data):
                # Check if we can convert to integer
                if col_data.notna().all() and (col_data % 1 == 0).all():
                    df[col] = col_data.astype('int64')
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(col_data, downcast='float')
        
        return df
    
    def _optimize_string_columns(self, df: pd.DataFrame, aggressive: bool) -> pd.DataFrame:
        """Optimize string columns using categories."""
        string_columns = df.select_dtypes(include=['object']).columns
        
        for col in string_columns:
            col_data = df[col]
            
            # Skip if mostly NaN
            if col_data.isna().sum() / len(col_data) > 0.9:
                continue
            
            # Check if column should be categorical
            unique_ratio = col_data.nunique() / len(col_data)
            
            # Convert to category if low cardinality or aggressive mode
            if unique_ratio < 0.5 or (aggressive and unique_ratio < 0.8):
                try:
                    df[col] = col_data.astype('category')
                except Exception as e:
                    self.logger.warning(f"Failed to convert {col} to category: {e}")
        
        return df
    
    def _optimize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize datetime columns."""
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            # Convert to most efficient datetime type
            try:
                # Use datetime64[ns] only if necessary
                col_data = df[col]
                min_date = col_data.min()
                max_date = col_data.max()
                
                # If date range is small, use lower precision
                date_range = (max_date - min_date).days
                if date_range < 365:  # Less than 1 year
                    df[col] = col_data.astype('datetime64[D]')
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize datetime column {col}: {e}")
        
        return df
    
    def _reduce_precision(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce precision for float columns where appropriate."""
        float_columns = df.select_dtypes(include=['float']).columns
        
        for col in float_columns:
            # Skip critical financial columns
            if any(keyword in col.lower() for keyword in ['price', 'value', 'pnl', 'balance']):
                continue
            
            col_data = df[col]
            
            # Check if we can reduce precision without significant loss
            if col_data.notna().any():
                # Round to 4 decimal places and check if values are preserved
                rounded = col_data.round(4)
                relative_error = abs((col_data - rounded) / col_data).max()
                
                if relative_error < 1e-6:  # Less than 0.0001% error
                    df[col] = rounded.astype('float32')
        
        return df

class ChunkedDataProcessor:
    """Process large datasets in chunks to manage memory."""
    
    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    async def process_large_dataset(self, data_source, processing_func, **kwargs):
        """Process large dataset in chunks with memory management."""
        results = []
        chunk_count = 0
        
        for chunk in self._get_chunks(data_source):
            chunk_count += 1
            
            # Monitor memory usage
            memory_before = psutil.Process().memory_info().rss / 1024**2
            
            try:
                # Process chunk
                chunk_result = await processing_func(chunk, **kwargs)
                results.append(chunk_result)
                
                # Log progress every 10 chunks
                if chunk_count % 10 == 0:
                    memory_after = psutil.Process().memory_info().rss / 1024**2
                    self.logger.info(f"Processed chunk {chunk_count}, "
                                   f"Memory: {memory_after:.1f}MB "
                                   f"(+{memory_after-memory_before:.1f}MB)")
                
                # Force garbage collection periodically
                if chunk_count % 5 == 0:
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_count}: {e}")
                continue
        
        return results
    
    def _get_chunks(self, data_source):
        """Generate chunks from data source."""
        if isinstance(data_source, pd.DataFrame):
            # DataFrame chunking
            for i in range(0, len(data_source), self.chunk_size):
                yield data_source.iloc[i:i + self.chunk_size]
        
        elif hasattr(data_source, '__iter__'):
            # Iterable chunking
            chunk = []
            for item in data_source:
                chunk.append(item)
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
            
            if chunk:  # Yield remaining items
                yield chunk

class MemoryMonitor:
    """Monitor memory usage and provide alerts."""
    
    def __init__(self, warning_threshold_mb: int = 2048, critical_threshold_mb: int = 4096):
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.logger = logging.getLogger(__name__)
        self._last_warning = 0
    
    def check_memory_usage(self, operation_name: str = "operation") -> Dict[str, Any]:
        """Check current memory usage and alert if necessary."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024**2
        memory_percent = process.memory_percent()
        
        current_time = time.time()
        
        # Alert levels
        if memory_mb > self.critical_threshold:
            self.logger.critical(f"CRITICAL: Memory usage {memory_mb:.1f}MB "
                               f"({memory_percent:.1f}%) during {operation_name}")
            # Force garbage collection
            gc.collect()
            
        elif memory_mb > self.warning_threshold:
            # Rate limit warnings to avoid spam
            if current_time - self._last_warning > 60:  # 1 minute
                self.logger.warning(f"HIGH: Memory usage {memory_mb:.1f}MB "
                                  f"({memory_percent:.1f}%) during {operation_name}")
                self._last_warning = current_time
        
        return {
            'memory_mb': memory_mb,
            'memory_percent': memory_percent,
            'warning_level': 'critical' if memory_mb > self.critical_threshold
                           else 'warning' if memory_mb > self.warning_threshold
                           else 'normal'
        }
    
    def log_memory_usage(self, context: str):
        """Decorator to log memory usage around operations."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Before
                before = self.check_memory_usage(f"{context} (start)")
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    # After
                    after = self.check_memory_usage(f"{context} (end)")
                    memory_diff = after['memory_mb'] - before['memory_mb']
                    
                    if abs(memory_diff) > 100:  # Log if > 100MB change
                        self.logger.info(f"{context}: Memory changed by {memory_diff:+.1f}MB")
            
            return wrapper
        return decorator

# Enhanced DataFrame operations with memory awareness
class MemoryEfficientDataFrame:
    """Wrapper for DataFrame operations with memory optimization."""
    
    def __init__(self, df: pd.DataFrame, auto_optimize: bool = True):
        self.optimizer = AdvancedMemoryOptimizer()
        self.monitor = MemoryMonitor()
        
        if auto_optimize:
            report = self.optimizer.optimize_trading_dataframe(df)
            self.df = df  # Use optimized version
            self.optimization_report = report
        else:
            self.df = df
            self.optimization_report = None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        memory_usage = self.df.memory_usage(deep=True)
        
        return {
            'total_mb': memory_usage.sum() / 1024**2,
            'per_column_mb': {col: usage / 1024**2 
                             for col, usage in memory_usage.items()},
            'row_count': len(self.df),
            'column_count': len(self.df.columns)
        }
    
    def apply_operation(self, operation_func, *args, **kwargs):
        """Apply operation with memory monitoring."""
        with self.monitor.log_memory_usage(f"DataFrame operation"):
            return operation_func(self.df, *args, **kwargs)

# Usage examples
def create_memory_efficient_pipeline():
    """Example of memory-efficient data processing pipeline."""
    
    # Initialize components
    optimizer = AdvancedMemoryOptimizer()
    processor = ChunkedDataProcessor(chunk_size=10000)
    monitor = MemoryMonitor()
    
    async def process_trading_data(file_path: str):
        """Process large trading data file efficiently."""
        
        # Read data in chunks
        chunk_reader = pd.read_csv(file_path, chunksize=10000)
        
        processed_chunks = []
        
        for chunk_num, chunk in enumerate(chunk_reader):
            # Optimize each chunk
            report = optimizer.optimize_trading_dataframe(chunk, aggressive=True)
            
            # Monitor memory
            memory_status = monitor.check_memory_usage(f"chunk_{chunk_num}")
            
            # Process chunk (your trading logic here)
            processed_chunk = await process_chunk_logic(chunk)
            processed_chunks.append(processed_chunk)
            
            # Cleanup
            del chunk
            if chunk_num % 5 == 0:
                gc.collect()
        
        # Combine results
        final_result = pd.concat(processed_chunks, ignore_index=True)
        
        # Final optimization
        final_report = optimizer.optimize_trading_dataframe(final_result)
        
        return MemoryEfficientDataFrame(final_result)

async def process_chunk_logic(chunk: pd.DataFrame) -> pd.DataFrame:
    """Your trading-specific chunk processing logic."""
    # Calculate indicators
    chunk['sma_20'] = chunk['close'].rolling(20).mean()
    chunk['rsi'] = calculate_rsi(chunk['close'])
    
    # Filter relevant data
    return chunk[['timestamp', 'symbol', 'close', 'sma_20', 'rsi']]
```

Test memory optimization:
```python
async def test_memory_optimization():
    """Test memory optimization with realistic trading data."""
    # Generate test data
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=100000, freq='1min'),
        'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], 100000),
        'open': np.random.uniform(40000, 60000, 100000),
        'high': np.random.uniform(40000, 60000, 100000),
        'low': np.random.uniform(40000, 60000, 100000),
        'close': np.random.uniform(40000, 60000, 100000),
        'volume': np.random.uniform(100, 10000, 100000),
    }
    
    df = pd.DataFrame(data)
    
    # Test optimization
    optimizer = AdvancedMemoryOptimizer()
    report = optimizer.optimize_trading_dataframe(df, aggressive=True)
    
    print(f"Memory reduction: {report.reduction_percent:.1f}%")
    print(f"Optimized dtypes: {report.dtypes_optimized}")
    
    assert report.reduction_percent > 50  # Should achieve >50% reduction
```
```

## ⚡ Medium Priority Algorithm Optimizations (Week 3)

### Prompt 7: Implement Ultra-High-Performance Indicators with Numba JIT
```
Replace pandas-based indicator calculations with Numba-optimized versions for maximum performance.

Create `indicators/numba_indicators.py`:

```python
import numba
import numpy as np
import math
from typing import Tuple, Optional
from numba import jit, prange, types
from numba.typed import List as NumbaList
import logging

# Configure Numba for optimal trading performance
numba.config.THREADING_LAYER = 'tbb'  # Use Intel TBB for better performance
numba.config.NUMBA_NUM_THREADS = 4    # Optimize for typical trading server

@jit(nopython=True, cache=True, fastmath=True)
def fast_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Ultra-fast Simple Moving Average using Numba JIT."""
    n = len(prices)
    result = np.full(n, np.nan)
    
    if window > n:
        return result
    
    # Calculate first SMA
    window_sum = 0.0
    for i in range(window):
        window_sum += prices[i]
    result[window - 1] = window_sum / window
    
    # Rolling calculation for efficiency
    for i in range(window, n):
        window_sum = window_sum - prices[i - window] + prices[i]
        result[i] = window_sum / window
    
    return result

@jit(nopython=True, cache=True, fastmath=True)
def fast_ema(prices: np.ndarray, window: int, adjust: bool = True) -> np.ndarray:
    """Ultra-fast Exponential Moving Average with optional adjustment."""
    n = len(prices)
    result = np.full(n, np.nan)
    
    if n == 0 or window <= 0:
        return result
    
    alpha = 2.0 / (window + 1.0)
    
    if adjust:
        # Pandas-compatible EMA calculation
        result[0] = prices[0]
        for i in range(1, n):
            if not np.isnan(prices[i]):
                if np.isnan(result[i-1]):
                    result[i] = prices[i]
                else:
                    result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    else:
        # Simple EMA
        result[0] = prices[0]
        for i in range(1, n):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    
    return result

@jit(nopython=True, cache=True, fastmath=True)
def fast_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """Ultra-fast RSI calculation using optimized gain/loss computation."""
    n = len(prices)
    result = np.full(n, np.nan)
    
    if n < window + 1:
        return result
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    # Calculate first RSI
    if avg_loss == 0:
        result[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[window] = 100.0 - (100.0 / (1.0 + rs))
    
    # Rolling calculation using Wilder's smoothing
    alpha = 1.0 / window
    for i in range(window + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        
        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result

@jit(nopython=True, cache=True, fastmath=True)
def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
             window: int = 14) -> np.ndarray:
    """Ultra-fast Average True Range calculation."""
    n = len(high)
    result = np.full(n, np.nan)
    
    if n < 2:
        return result
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]  # First TR is just high - low
    
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_cp = abs(high[i] - close[i-1])
        l_cp = abs(low[i] - close[i-1])
        tr[i] = max(h_l, max(h_cp, l_cp))
    
    # Calculate ATR using RMA (same as EMA with alpha = 1/period)
    alpha = 1.0 / window
    atr = tr[0]
    result[0] = atr
    
    for i in range(1, n):
        atr = alpha * tr[i] + (1 - alpha) * atr
        result[i] = atr
    
    return result

@jit(nopython=True, cache=True, fastmath=True)
def fast_macd(prices: np.ndarray, fast_period: int = 12, 
              slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast MACD calculation returning (macd_line, signal_line, histogram)."""
    # Calculate EMAs
    fast_ema = fast_ema(prices, fast_period)
    slow_ema = fast_ema(prices, slow_period)
    
    # MACD line
    macd_line = fast_ema - slow_ema
    
    # Signal line (EMA of MACD)
    signal_line = fast_ema(macd_line, signal_period)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

@jit(nopython=True, cache=True, fastmath=True)
def fast_bollinger_bands(prices: np.ndarray, window: int = 20, 
                        num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast Bollinger Bands calculation."""
    n = len(prices)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if window > n:
        return upper, middle, lower
    
    # Rolling mean and std calculation
    for i in range(window - 1, n):
        window_data = prices[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        
        middle[i] = mean_val
        upper[i] = mean_val + num_std * std_val
        lower[i] = mean_val - num_std * std_val
    
    return upper, middle, lower

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def fast_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Ultra-fast Stochastic Oscillator with parallel computation."""
    n = len(high)
    k_values = np.full(n, np.nan)
    
    # Calculate %K
    for i in prange(k_period - 1, n):
        period_high = high[i - k_period + 1:i + 1].max()
        period_low = low[i - k_period + 1:i + 1].min()
        
        if period_high != period_low:
            k_values[i] = 100.0 * (close[i] - period_low) / (period_high - period_low)
        else:
            k_values[i] = 50.0
    
    # Calculate %D (SMA of %K)
    d_values = fast_sma(k_values, d_period)
    
    return k_values, d_values

@jit(nopython=True, cache=True, fastmath=True)
def fast_vwap(prices: np.ndarray, volumes: np.ndarray, window: int) -> np.ndarray:
    """Volume Weighted Average Price with rolling window."""
    n = len(prices)
    result = np.full(n, np.nan)
    
    if window > n:
        return result
    
    for i in range(window - 1, n):
        period_prices = prices[i - window + 1:i + 1]
        period_volumes = volumes[i - window + 1:i + 1]
        
        total_volume = np.sum(period_volumes)
        if total_volume > 0:
            vwap = np.sum(period_prices * period_volumes) / total_volume
            result[i] = vwap
    
    return result

@jit(nopython=True, cache=True, fastmath=True)
def fast_williamsr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   window: int = 14) -> np.ndarray:
    """Williams %R calculation."""
    n = len(high)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        period_high = high[i - window + 1:i + 1].max()
        period_low = low[i - window + 1:i + 1].min()
        
        if period_high != period_low:
            result[i] = -100.0 * (period_high - close[i]) / (period_high - period_low)
        else:
            result[i] = -50.0
    
    return result

# Batch processing for multiple indicators
@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def calculate_all_indicators(prices: np.ndarray, high: np.ndarray, 
                           low: np.ndarray, volume: np.ndarray) -> dict:
    """Calculate multiple indicators in parallel for maximum efficiency."""
    # Note: numba doesn't support returning dicts directly
    # This function would need to be restructured to return arrays
    
    # Common parameters
    sma_20 = fast_sma(prices, 20)
    ema_12 = fast_ema(prices, 12)
    ema_26 = fast_ema(prices, 26)
    rsi_14 = fast_rsi(prices, 14)
    atr_14 = fast_atr(high, low, prices, 14)
    
    return sma_20, ema_12, ema_26, rsi_14, atr_14

# Wrapper class for easier integration
class NumbaIndicatorEngine:
    """High-performance indicator engine using Numba JIT compilation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compilation_cache = {}
    
    def calculate_indicators(self, price_data: dict) -> dict:
        """Calculate all indicators with maximum performance."""
        try:
            # Extract arrays
            close = np.array(price_data['close'], dtype=np.float64)
            high = np.array(price_data.get('high', close), dtype=np.float64)
            low = np.array(price_data.get('low', close), dtype=np.float64)
            volume = np.array(price_data.get('volume', np.ones_like(close)), dtype=np.float64)
            
            # Calculate indicators
            indicators = {
                'sma_20': fast_sma(close, 20),
                'sma_50': fast_sma(close, 50),
                'ema_12': fast_ema(close, 12),
                'ema_26': fast_ema(close, 26),
                'rsi_14': fast_rsi(close, 14),
                'atr_14': fast_atr(high, low, close, 14),
                'vwap_20': fast_vwap(close, volume, 20)
            }
            
            # MACD
            macd, signal, histogram = fast_macd(close)
            indicators.update({
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': histogram
            })
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = fast_bollinger_bands(close)
            indicators.update({
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower
            })
            
            # Stochastic
            stoch_k, stoch_d = fast_stochastic(high, low, close)
            indicators.update({
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            })
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error: {e}")
            return {}
    
    def benchmark_performance(self, data_size: int = 10000) -> dict:
        """Benchmark indicator calculation performance."""
        import time
        
        # Generate test data
        np.random.seed(42)
        prices = np.random.randn(data_size).cumsum() + 100
        high = prices + np.random.uniform(0, 2, data_size)
        low = prices - np.random.uniform(0, 2, data_size)
        volume = np.random.uniform(1000, 10000, data_size)
        
        price_data = {
            'close': prices,
            'high': high,
            'low': low,
            'volume': volume
        }
        
        # Warmup (trigger JIT compilation)
        self.calculate_indicators(price_data)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(100):  # 100 iterations
            indicators = self.calculate_indicators(price_data)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 100
        throughput = data_size / avg_time
        
        return {
            'data_size': data_size,
            'avg_calculation_time_ms': avg_time * 1000,
            'throughput_points_per_second': throughput,
            'indicators_calculated': len(indicators)
        }

# Integration with existing codebase
async def upgrade_indicator_system():
    """Upgrade existing indicator system to use Numba."""
    
    # Initialize high-performance engine
    engine = NumbaIndicatorEngine()
    
    # Benchmark performance
    benchmark = engine.benchmark_performance(10000)
    print(f"Numba indicators: {benchmark['avg_calculation_time_ms']:.2f}ms per calculation")
    print(f"Throughput: {benchmark['throughput_points_per_second']:,.0f} points/second")
    
    return engine
```

Update existing `indicators.py` to use Numba versions:
```python
class EnhancedIndicators:
    def __init__(self):
        self.numba_engine = NumbaIndicatorEngine()
        self.logger = logging.getLogger(__name__)
    
    async def calculate_indicators_async(self, symbol: str, price_data: dict) -> dict:
        """Calculate indicators asynchronously using thread pool."""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive calculation in thread pool
        indicators = await loop.run_in_executor(
            None, 
            self.numba_engine.calculate_indicators, 
            price_data
        )
        
        self.logger.debug(f"Calculated {len(indicators)} indicators for {symbol}")
        return indicators

# Performance comparison
async def compare_indicator_performance():
    """Compare old vs new indicator performance."""
    import time
    
    # Generate test data
    np.random.seed(42)
    data_size = 10000
    prices = np.random.randn(data_size).cumsum() + 100
    
    price_data = {
        'close': prices,
        'high': prices + np.random.uniform(0, 2, data_size),
        'low': prices - np.random.uniform(0, 2, data_size),
        'volume': np.random.uniform(1000, 10000, data_size)
    }
    
    # Test old pandas-based method
    df = pd.DataFrame(price_data)
    start_time = time.perf_counter()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi_14'] = calculate_rsi_pandas(df['close'])  # Your existing function
    pandas_time = time.perf_counter() - start_time
    
    # Test new Numba method
    engine = NumbaIndicatorEngine()
    # Warmup
    engine.calculate_indicators(price_data)
    
    start_time = time.perf_counter()
    numba_indicators = engine.calculate_indicators(price_data)
    numba_time = time.perf_counter() - start_time
    
    speedup = pandas_time / numba_time
    print(f"Pandas method: {pandas_time*1000:.2f}ms")
    print(f"Numba method: {numba_time*1000:.2f}ms")
    print(f"Speedup: {speedup:.1f}x faster")
    
    assert speedup > 10  # Should be at least 10x faster
```

Test the implementation:
```python
async def test_numba_indicators():
    """Test Numba indicator accuracy and performance."""
    
    # Test data
    prices = np.array([100, 101, 102, 99, 98, 103, 105, 104, 106, 107])
    
    # Test SMA
    sma = fast_sma(prices, 3)
    expected_sma_last = np.mean(prices[-3:])  # Last 3 values
    assert abs(sma[-1] - expected_sma_last) < 1e-10
    
    # Test RSI bounds
    rsi = fast_rsi(prices, 14)
    valid_rsi = rsi[~np.isnan(rsi)]
    assert np.all(valid_rsi >= 0) and np.all(valid_rsi <= 100)
    
    # Performance benchmark
    engine = NumbaIndicatorEngine()
    benchmark = engine.benchmark_performance(10000)
    
    assert benchmark['avg_calculation_time_ms'] < 10  # Should be under 10ms
    print(f"✓ Numba indicators: {benchmark['avg_calculation_time_ms']:.2f}ms")
```
```

### Prompt 8: Implement Advanced Smart Order Execution System
```
Create an intelligent order routing and execution system with market microstructure awareness.

Create `execution/smart_execution_engine.py`:

```python
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"
    ADAPTIVE = "ADAPTIVE"

class VenueType(Enum):
    PRIMARY = "PRIMARY"
    DARK_POOL = "DARK_POOL"
    ECN = "ECN"

@dataclass
class VenueInfo:
    venue_id: str
    venue_type: VenueType
    fees: float  # bps
    avg_fill_rate: float  # percentage
    latency_ms: float
    min_quantity: float
    max_quantity: float
    supported_order_types: List[OrderType]

@dataclass
class MarketMicrostructure:
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume_1min: float
    volume_5min: float
    volatility: float
    tick_size: float
    timestamp: datetime

@dataclass
class TradingSignal:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    signal_strength: float  # 0-1
    urgency: float  # 0-1
    max_participation_rate: float = 0.2  # Max % of volume
    time_limit: Optional[timedelta] = None

@dataclass
class ExecutionResult:
    fill_price: float
    fill_quantity: float
    venue: str
    timestamp: datetime
    slippage_bps: float
    implementation_shortfall_bps: float
    market_impact_bps: float

class SmartExecutionEngine:
    """Advanced execution engine with intelligent order routing."""
    
    def __init__(self):
        self.venues = self._initialize_venues()
        self.logger = logging.getLogger(__name__)
        self._execution_history = []
        self._market_impact_model = MarketImpactModel()
        self._adaptive_algo = AdaptiveAlgorithm()
    
    def _initialize_venues(self) -> Dict[str, VenueInfo]:
        """Initialize available trading venues."""
        return {
            'BINANCE': VenueInfo(
                venue_id='BINANCE',
                venue_type=VenueType.PRIMARY,
                fees=0.1,  # 0.1 bps
                avg_fill_rate=0.95,
                latency_ms=50,
                min_quantity=0.001,
                max_quantity=1000000,
                supported_order_types=[OrderType.MARKET, OrderType.LIMIT, OrderType.ICEBERG]
            ),
            'COINBASE': VenueInfo(
                venue_id='COINBASE',
                venue_type=VenueType.ECN,
                fees=0.5,
                avg_fill_rate=0.88,
                latency_ms=80,
                min_quantity=0.01,
                max_quantity=500000,
                supported_order_types=[OrderType.MARKET, OrderType.LIMIT]
            ),
            'DARK_POOL_1': VenueInfo(
                venue_id='DARK_POOL_1',
                venue_type=VenueType.DARK_POOL,
                fees=0.2,
                avg_fill_rate=0.70,
                latency_ms=100,
                min_quantity=1.0,
                max_quantity=100000,
                supported_order_types=[OrderType.LIMIT, OrderType.ICEBERG]
            )
        }
    
    async def execute_signal(self, signal: TradingSignal, 
                           market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute trading signal with optimal strategy selection."""
        
        # 1. Analyze market conditions
        market_conditions = self._analyze_market_conditions(market_data)
        
        # 2. Select optimal execution strategy
        strategy = self._select_execution_strategy(signal, market_conditions)
        
        # 3. Route to best venues
        venue_allocations = await self._smart_venue_routing(signal, market_data)
        
        # 4. Execute with selected strategy
        execution_results = await self._execute_with_strategy(
            signal, strategy, venue_allocations, market_data
        )
        
        # 5. Track performance
        self._track_execution_performance(signal, execution_results, market_data)
        
        return execution_results
    
    def _analyze_market_conditions(self, market_data: MarketMicrostructure) -> Dict:
        """Analyze current market microstructure conditions."""
        spread_bps = ((market_data.ask_price - market_data.bid_price) / 
                     market_data.last_price) * 10000
        
        # Market quality indicators
        conditions = {
            'spread_bps': spread_bps,
            'market_depth': min(market_data.bid_size, market_data.ask_size),
            'volume_intensity': market_data.volume_1min / max(market_data.volume_5min/5, 1),
            'volatility_regime': 'high' if market_data.volatility > 0.02 else 'normal',
            'liquidity_score': self._calculate_liquidity_score(market_data),
            'market_impact_estimate': self._market_impact_model.estimate_impact(market_data)
        }
        
        return conditions
    
    def _calculate_liquidity_score(self, market_data: MarketMicrostructure) -> float:
        """Calculate market liquidity score (0-1)."""
        # Combine spread, depth, and volume
        spread_component = max(0, 1 - (market_data.ask_price - market_data.bid_price) / 
                              market_data.last_price * 1000)
        depth_component = min(1, (market_data.bid_size + market_data.ask_size) / 100)
        volume_component = min(1, market_data.volume_1min / 1000)
        
        return (spread_component * 0.4 + depth_component * 0.3 + volume_component * 0.3)
    
    def _select_execution_strategy(self, signal: TradingSignal, 
                                 conditions: Dict) -> OrderType:
        """Select optimal execution strategy based on conditions."""
        
        # High urgency + good liquidity = Market order
        if signal.urgency > 0.8 and conditions['liquidity_score'] > 0.7:
            return OrderType.MARKET
        
        # Large order + low urgency = TWAP
        if signal.quantity > 1000 and signal.urgency < 0.3:
            return OrderType.TWAP
        
        # High volatility = Adaptive algorithm
        if conditions['volatility_regime'] == 'high':
            return OrderType.ADAPTIVE
        
        # Large order + need to hide size = Iceberg
        if signal.quantity > 500 and conditions['market_depth'] < signal.quantity * 0.5:
            return OrderType.ICEBERG
        
        # Default to VWAP for balanced execution
        return OrderType.VWAP
    
    async def _smart_venue_routing(self, signal: TradingSignal, 
                                 market_data: MarketMicrostructure) -> Dict[str, float]:
        """Determine optimal venue allocation for the order."""
        
        venue_scores = {}
        
        for venue_id, venue_info in self.venues.items():
            # Calculate venue score based on multiple factors
            score = self._calculate_venue_score(venue_info, signal, market_data)
            
            if score > 0.3:  # Minimum threshold
                venue_scores[venue_id] = score
        
        # Normalize to allocations that sum to 1
        total_score = sum(venue_scores.values())
        allocations = {venue: score/total_score 
                      for venue, score in venue_scores.items()}
        
        # Apply quantity constraints
        allocations = self._apply_venue_constraints(allocations, signal)
        
        return allocations
    
    def _calculate_venue_score(self, venue: VenueInfo, signal: TradingSignal,
                             market_data: MarketMicrostructure) -> float:
        """Calculate score for routing to a specific venue."""
        
        # Base score from fill rate
        score = venue.avg_fill_rate
        
        # Adjust for fees (lower is better)
        fee_penalty = venue.fees / 100  # Convert bps to percentage
        score *= (1 - fee_penalty)
        
        # Adjust for latency (lower is better for urgent orders)
        if signal.urgency > 0.5:
            latency_penalty = venue.latency_ms / 1000  # Convert to seconds
            score *= (1 - latency_penalty * signal.urgency)
        
        # Venue type preferences
        if venue.venue_type == VenueType.DARK_POOL and signal.quantity > 100:
            score *= 1.2  # Prefer dark pools for large orders
        
        # Size constraints
        if signal.quantity < venue.min_quantity or signal.quantity > venue.max_quantity:
            score = 0
        
        return score
    
    def _apply_venue_constraints(self, allocations: Dict[str, float], 
                               signal: TradingSignal) -> Dict[str, float]:
        """Apply venue-specific quantity constraints."""
        
        constrained_allocations = {}
        remaining_quantity = signal.quantity
        
        # Sort venues by allocation percentage (highest first)
        sorted_venues = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
        
        for venue_id, allocation in sorted_venues:
            venue = self.venues[venue_id]
            allocated_quantity = signal.quantity * allocation
            
            # Apply venue limits
            max_venue_quantity = min(venue.max_quantity, remaining_quantity)
            final_quantity = min(allocated_quantity, max_venue_quantity)
            
            if final_quantity >= venue.min_quantity:
                constrained_allocations[venue_id] = final_quantity / signal.quantity
                remaining_quantity -= final_quantity
        
        return constrained_allocations
    
    async def _execute_with_strategy(self, signal: TradingSignal, strategy: OrderType,
                                   allocations: Dict[str, float], 
                                   market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute order using selected strategy and venue allocations."""
        
        execution_results = []
        
        if strategy == OrderType.MARKET:
            results = await self._execute_market_orders(signal, allocations, market_data)
            execution_results.extend(results)
            
        elif strategy == OrderType.TWAP:
            results = await self._execute_twap(signal, allocations, market_data)
            execution_results.extend(results)
            
        elif strategy == OrderType.VWAP:
            results = await self._execute_vwap(signal, allocations, market_data)
            execution_results.extend(results)
            
        elif strategy == OrderType.ICEBERG:
            results = await self._execute_iceberg(signal, allocations, market_data)
            execution_results.extend(results)
            
        elif strategy == OrderType.ADAPTIVE:
            results = await self._execute_adaptive(signal, allocations, market_data)
            execution_results.extend(results)
        
        return execution_results
    
    async def _execute_market_orders(self, signal: TradingSignal, 
                                   allocations: Dict[str, float],
                                   market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute immediate market orders across venues."""
        
        results = []
        
        for venue_id, allocation in allocations.items():
            quantity = signal.quantity * allocation
            
            # Estimate fill price with slippage
            if signal.side == 'BUY':
                estimated_price = market_data.ask_price * (1 + self._estimate_slippage(quantity, market_data))
            else:
                estimated_price = market_data.bid_price * (1 - self._estimate_slippage(quantity, market_data))
            
            # Simulate execution (replace with actual API calls)
            fill_result = await self._simulate_execution(
                venue_id, quantity, estimated_price, market_data
            )
            
            results.append(fill_result)
        
        return results
    
    async def _execute_twap(self, signal: TradingSignal, allocations: Dict[str, float],
                          market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute Time-Weighted Average Price strategy."""
        
        time_horizon = signal.time_limit or timedelta(minutes=30)
        num_slices = min(10, int(time_horizon.total_seconds() / 60))  # Max 10 slices
        slice_quantity = signal.quantity / num_slices
        slice_interval = time_horizon.total_seconds() / num_slices
        
        results = []
        
        for i in range(num_slices):
            # Wait for next slice time
            if i > 0:
                await asyncio.sleep(slice_interval)
            
            # Execute slice across venues
            for venue_id, allocation in allocations.items():
                venue_quantity = slice_quantity * allocation
                
                # Use limit orders at mid-price for TWAP
                limit_price = (market_data.bid_price + market_data.ask_price) / 2
                
                fill_result = await self._simulate_execution(
                    venue_id, venue_quantity, limit_price, market_data
                )
                
                results.append(fill_result)
        
        return results
    
    async def _execute_vwap(self, signal: TradingSignal, allocations: Dict[str, float],
                          market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute Volume-Weighted Average Price strategy."""
        
        # Estimate volume profile for the day
        hourly_volume_profile = self._get_volume_profile(signal.symbol)
        
        # Calculate participation rate
        target_participation = min(signal.max_participation_rate, 0.2)
        
        results = []
        total_executed = 0
        
        while total_executed < signal.quantity:
            # Calculate slice size based on current volume
            current_volume_rate = market_data.volume_1min
            slice_quantity = min(
                current_volume_rate * target_participation,
                signal.quantity - total_executed
            )
            
            # Execute slice
            for venue_id, allocation in allocations.items():
                venue_quantity = slice_quantity * allocation
                
                # Price slightly better than mid to ensure fills
                mid_price = (market_data.bid_price + market_data.ask_price) / 2
                if signal.side == 'BUY':
                    limit_price = mid_price + market_data.tick_size
                else:
                    limit_price = mid_price - market_data.tick_size
                
                fill_result = await self._simulate_execution(
                    venue_id, venue_quantity, limit_price, market_data
                )
                
                results.append(fill_result)
                total_executed += fill_result.fill_quantity
            
            # Wait before next slice
            await asyncio.sleep(60)  # 1 minute intervals
        
        return results
    
    async def _execute_iceberg(self, signal: TradingSignal, allocations: Dict[str, float],
                             market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute iceberg orders to hide large size."""
        
        # Iceberg slice size (typically 5-10% of total order)
        slice_size = signal.quantity * 0.08
        
        results = []
        remaining_quantity = signal.quantity
        
        while remaining_quantity > 0:
            current_slice = min(slice_size, remaining_quantity)
            
            # Execute current slice
            for venue_id, allocation in allocations.items():
                venue_quantity = current_slice * allocation
                
                # Use limit orders at favorable prices
                if signal.side == 'BUY':
                    limit_price = market_data.bid_price + market_data.tick_size
                else:
                    limit_price = market_data.ask_price - market_data.tick_size
                
                fill_result = await self._simulate_execution(
                    venue_id, venue_quantity, limit_price, market_data
                )
                
                results.append(fill_result)
                remaining_quantity -= fill_result.fill_quantity
            
            # Wait between slices to avoid detection
            await asyncio.sleep(30)
        
        return results
    
    async def _execute_adaptive(self, signal: TradingSignal, allocations: Dict[str, float],
                              market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute using adaptive algorithm that adjusts to market conditions."""
        
        return await self._adaptive_algo.execute(signal, allocations, market_data, self)
    
    def _estimate_slippage(self, quantity: float, market_data: MarketMicrostructure) -> float:
        """Estimate slippage based on order size and market depth."""
        
        # Simple linear impact model (replace with more sophisticated model)
        market_depth = min(market_data.bid_size, market_data.ask_size)
        impact_ratio = quantity / max(market_depth, 1)
        
        # Base slippage increases with size
        base_slippage = 0.0001  # 1 bps
        size_impact = impact_ratio * 0.001  # Additional impact
        
        return base_slippage + size_impact
    
    async def _simulate_execution(self, venue_id: str, quantity: float, 
                                price: float, market_data: MarketMicrostructure) -> ExecutionResult:
        """Simulate order execution (replace with actual venue API calls)."""
        
        venue = self.venues[venue_id]
        
        # Simulate partial fills and latency
        await asyncio.sleep(venue.latency_ms / 1000)
        
        # Probability of fill based on venue characteristics
        fill_probability = venue.avg_fill_rate
        fill_quantity = quantity * fill_probability
        
        # Simulate slippage
        slippage = self._estimate_slippage(quantity, market_data)
        if price > market_data.last_price:  # Buy order
            fill_price = price * (1 + slippage)
        else:  # Sell order
            fill_price = price * (1 - slippage)
        
        # Calculate performance metrics
        slippage_bps = abs(fill_price - market_data.last_price) / market_data.last_price * 10000
        
        return ExecutionResult(
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            venue=venue_id,
            timestamp=datetime.utcnow(),
            slippage_bps=slippage_bps,
            implementation_shortfall_bps=slippage_bps,  # Simplified
            market_impact_bps=slippage_bps * 0.7  # Estimate
        )
    
    def _get_volume_profile(self, symbol: str) -> np.ndarray:
        """Get historical volume profile for VWAP calculation."""
        # Simplified - return uniform profile
        # In practice, load from historical data
        return np.ones(24) / 24  # Uniform hourly distribution
    
    def _track_execution_performance(self, signal: TradingSignal, 
                                   results: List[ExecutionResult],
                                   market_data: MarketMicrostructure):
        """Track execution performance for algorithm improvement."""
        
        total_quantity = sum(r.fill_quantity for r in results)
        weighted_fill_price = sum(r.fill_price * r.fill_quantity for r in results) / total_quantity
        avg_slippage = sum(r.slippage_bps * r.fill_quantity for r in results) / total_quantity
        
        performance_metrics = {
            'signal_id': id(signal),
            'symbol': signal.symbol,
            'total_quantity': total_quantity,
            'fill_rate': total_quantity / signal.quantity,
            'weighted_avg_price': weighted_fill_price,
            'avg_slippage_bps': avg_slippage,
            'num_venues': len(set(r.venue for r in results)),
            'execution_time': (results[-1].timestamp - results[0].timestamp).total_seconds(),
            'market_conditions': {
                'spread_bps': ((market_data.ask_price - market_data.bid_price) / 
                             market_data.last_price) * 10000,
                'volatility': market_data.volatility
            }
        }
        
        self._execution_history.append(performance_metrics)
        self.logger.info(f"Execution completed: {performance_metrics}")

class MarketImpactModel:
    """Model for estimating market impact of trades."""
    
    def __init__(self):
        # Simplified linear impact model parameters
        self.permanent_impact_coeff = 0.1
        self.temporary_impact_coeff = 0.5
        self.volume_adjustment = 0.01
    
    def estimate_impact(self, market_data: MarketMicrostructure, 
                       quantity: Optional[float] = None) -> Dict[str, float]:
        """Estimate market impact for given trade size."""
        
        if quantity is None:
            quantity = min(market_data.bid_size, market_data.ask_size) * 0.1
        
        # Volume participation rate
        participation_rate = quantity / max(market_data.volume_1min, 1)
        
        # Permanent impact (price moves permanently)
        permanent_impact = self.permanent_impact_coeff * np.sqrt(participation_rate)
        
        # Temporary impact (recovers after trade)
        temporary_impact = self.temporary_impact_coeff * participation_rate
        
        # Volatility adjustment
        vol_adjustment = market_data.volatility * self.volume_adjustment
        
        return {
            'permanent_impact_bps': permanent_impact * 10000 * (1 + vol_adjustment),
            'temporary_impact_bps': temporary_impact * 10000 * (1 + vol_adjustment),
            'total_impact_bps': (permanent_impact + temporary_impact) * 10000 * (1 + vol_adjustment)
        }

class AdaptiveAlgorithm:
    """Adaptive execution algorithm that learns from market conditions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.learning_rate = 0.01
        self.state_history = []
    
    async def execute(self, signal: TradingSignal, allocations: Dict[str, float],
                     market_data: MarketMicrostructure, 
                     engine: SmartExecutionEngine) -> List[ExecutionResult]:
        """Execute with adaptive strategy."""
        
        # Initialize adaptive parameters
        aggression_level = self._calculate_initial_aggression(signal, market_data)
        slice_size_ratio = 0.1  # Start with 10% slices
        
        results = []
        remaining_quantity = signal.quantity
        
        while remaining_quantity > 0:
            # Adapt strategy based on market feedback
            current_slice = remaining_quantity * slice_size_ratio
            
            # Execute current slice
            slice_results = []
            for venue_id, allocation in allocations.items():
                venue_quantity = current_slice * allocation
                
                # Adaptive pricing based on urgency and market conditions
                adaptive_price = self._calculate_adaptive_price(
                    signal, market_data, aggression_level
                )
                
                fill_result = await engine._simulate_execution(
                    venue_id, venue_quantity, adaptive_price, market_data
                )
                
                slice_results.append(fill_result)
            
            results.extend(slice_results)
            
            # Learn from execution results
            slice_performance = self._analyze_slice_performance(slice_results, market_data)
            aggression_level, slice_size_ratio = self._adapt_parameters(
                slice_performance, aggression_level, slice_size_ratio
            )
            
            # Update remaining quantity
            executed_in_slice = sum(r.fill_quantity for r in slice_results)
            remaining_quantity -= executed_in_slice
            
            # Adaptive wait time
            wait_time = self._calculate_wait_time(slice_performance, signal.urgency)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        return results
    
    def _calculate_initial_aggression(self, signal: TradingSignal, 
                                    market_data: MarketMicrostructure) -> float:
        """Calculate initial aggression level (0-1)."""
        # Combine signal urgency with market conditions
        base_aggression = signal.urgency
        
        # Adjust for volatility (less aggressive in high vol)
        vol_adjustment = max(0, 1 - market_data.volatility * 50)
        
        # Adjust for spread (less aggressive with wide spreads)
        spread_bps = ((market_data.ask_price - market_data.bid_price) / 
                     market_data.last_price) * 10000
        spread_adjustment = max(0, 1 - spread_bps / 100)
        
        return base_aggression * vol_adjustment * spread_adjustment
    
    def _calculate_adaptive_price(self, signal: TradingSignal, 
                                market_data: MarketMicrostructure, 
                                aggression: float) -> float:
        """Calculate adaptive limit price based on aggression level."""
        
        mid_price = (market_data.bid_price + market_data.ask_price) / 2
        half_spread = (market_data.ask_price - market_data.bid_price) / 2
        
        if signal.side == 'BUY':
            # More aggressive = closer to ask price
            return mid_price + half_spread * aggression
        else:
            # More aggressive = closer to bid price
            return mid_price - half_spread * aggression
    
    def _analyze_slice_performance(self, slice_results: List[ExecutionResult],
                                 market_data: MarketMicrostructure) -> Dict:
        """Analyze performance of the last slice execution."""
        
        if not slice_results:
            return {'fill_rate': 0, 'avg_slippage': 0, 'speed': 0}
        
        total_requested = sum(r.fill_quantity for r in slice_results)  # Approximation
        total_filled = sum(r.fill_quantity for r in slice_results)
        fill_rate = total_filled / max(total_requested, 1)
        
        avg_slippage = sum(r.slippage_bps for r in slice_results) / len(slice_results)
        
        # Speed score (fills per unit time)
        execution_duration = (slice_results[-1].timestamp - 
                            slice_results[0].timestamp).total_seconds()
        speed_score = len(slice_results) / max(execution_duration, 1)
        
        return {
            'fill_rate': fill_rate,
            'avg_slippage': avg_slippage,
            'speed_score': speed_score
        }
    
    def _adapt_parameters(self, performance: Dict, current_aggression: float,
                         current_slice_ratio: float) -> Tuple[float, float]:
        """Adapt algorithm parameters based on performance feedback."""
        
        # Adapt aggression based on fill rate and slippage
        if performance['fill_rate'] < 0.8:  # Low fill rate
            new_aggression = min(1.0, current_aggression + self.learning_rate)
        elif performance['avg_slippage'] > 10:  # High slippage
            new_aggression = max(0.1, current_aggression - self.learning_rate)
        else:
            new_aggression = current_aggression
        
        # Adapt slice size based on market response
        if performance['fill_rate'] > 0.9 and performance['avg_slippage'] < 5:
            # Can be more aggressive with slice size
            new_slice_ratio = min(0.3, current_slice_ratio * 1.1)
        elif performance['fill_rate'] < 0.5:
            # Reduce slice size
            new_slice_ratio = max(0.05, current_slice_ratio * 0.9)
        else:
            new_slice_ratio = current_slice_ratio
        
        return new_aggression, new_slice_ratio
    
    def _calculate_wait_time(self, performance: Dict, urgency: float) -> float:
        """Calculate wait time between slices."""
        
        base_wait = 30  # 30 seconds base
        
        # Reduce wait time for urgent orders
        urgency_adjustment = (1 - urgency) * 0.8
        
        # Increase wait time if experiencing high slippage
        slippage_adjustment = min(performance['avg_slippage'] / 20, 1.0)
        
        wait_time = base_wait * urgency_adjustment * (1 + slippage_adjustment)
        
        return max(5, wait_time)  # Minimum 5 seconds

# Integration with main trading system
class EnhancedExecutionEngine:
    """Enhanced execution engine integrating smart routing with existing system."""
    
    def __init__(self, existing_execution_engine):
        self.smart_engine = SmartExecutionEngine()
        self.legacy_engine = existing_execution_engine
        self.logger = logging.getLogger(__name__)
        self.performance_tracker = ExecutionPerformanceTracker()
    
    async def execute_trade(self, signal: TradingSignal, 
                           market_data: MarketMicrostructure,
                           use_smart_routing: bool = True) -> List[ExecutionResult]:
        """Execute trade with optional smart routing."""
        
        if use_smart_routing and self._should_use_smart_routing(signal, market_data):
            self.logger.info(f"Using smart execution for {signal.symbol}")
            results = await self.smart_engine.execute_signal(signal, market_data)
        else:
            self.logger.info(f"Using legacy execution for {signal.symbol}")
            results = await self._execute_with_legacy(signal, market_data)
        
        # Track performance for comparison
        await self.performance_tracker.record_execution(signal, results, market_data)
        
        return results
    
    def _should_use_smart_routing(self, signal: TradingSignal, 
                                market_data: MarketMicrostructure) -> bool:
        """Determine if smart routing should be used."""
        
        # Use smart routing for:
        # 1. Large orders
        # 2. High-value trades
        # 3. Volatile market conditions
        
        large_order = signal.quantity > 100
        high_value = signal.quantity * market_data.last_price > 10000
        volatile_market = market_data.volatility > 0.02
        
        return large_order or high_value or volatile_market
    
    async def _execute_with_legacy(self, signal: TradingSignal,
                                 market_data: MarketMicrostructure) -> List[ExecutionResult]:
        """Execute using legacy engine (placeholder)."""
        
        # Convert to legacy format and execute
        # This would integrate with your existing execution logic
        
        if signal.side == 'BUY':
            fill_price = market_data.ask_price
        else:
            fill_price = market_data.bid_price
        
        return [ExecutionResult(
            fill_price=fill_price,
            fill_quantity=signal.quantity,
            venue='LEGACY',
            timestamp=datetime.utcnow(),
            slippage_bps=5.0,  # Estimated
            implementation_shortfall_bps=5.0,
            market_impact_bps=3.0
        )]

class ExecutionPerformanceTracker:
    """Track and analyze execution performance over time."""
    
    def __init__(self):
        self.executions = []
        self.logger = logging.getLogger(__name__)
    
    async def record_execution(self, signal: TradingSignal, 
                             results: List[ExecutionResult],
                             market_data: MarketMicrostructure):
        """Record execution for performance analysis."""
        
        execution_record = {
            'timestamp': datetime.utcnow(),
            'symbol': signal.symbol,
            'side': signal.side,
            'requested_quantity': signal.quantity,
            'executed_quantity': sum(r.fill_quantity for r in results),
            'avg_price': sum(r.fill_price * r.fill_quantity for r in results) / 
                        sum(r.fill_quantity for r in results),
            'total_slippage_bps': sum(r.slippage_bps * r.fill_quantity for r in results) / 
                                sum(r.fill_quantity for r in results),
            'venues_used': list(set(r.venue for r in results)),
            'market_conditions': {
                'spread_bps': ((market_data.ask_price - market_data.bid_price) / 
                             market_data.last_price) * 10000,
                'volatility': market_data.volatility,
                'volume': market_data.volume_1min
            }
        }
        
        self.executions.append(execution_record)
        
        # Generate performance report periodically
        if len(self.executions) % 100 == 0:
            await self.generate_performance_report()
    
    async def generate_performance_report(self) -> Dict:
        """Generate comprehensive execution performance report."""
        
        if not self.executions:
            return {}
        
        recent_executions = self.executions[-100:]  # Last 100 executions
        
        report = {
            'total_executions': len(recent_executions),
            'avg_slippage_bps': np.mean([e['total_slippage_bps'] for e in recent_executions]),
            'fill_rate': np.mean([e['executed_quantity'] / e['requested_quantity'] 
                                for e in recent_executions]),
            'venue_distribution': self._calculate_venue_distribution(recent_executions),
            'performance_by_size': self._analyze_performance_by_size(recent_executions),
            'performance_by_volatility': self._analyze_performance_by_volatility(recent_executions)
        }
        
        self.logger.info(f"Execution Performance Report: {report}")
        return report
    
    def _calculate_venue_distribution(self, executions: List[Dict]) -> Dict:
        """Calculate distribution of executions across venues."""
        
        venue_counts = {}
        for execution in executions:
            for venue in execution['venues_used']:
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        total = sum(venue_counts.values())
        return {venue: count/total for venue, count in venue_counts.items()}
    
    def _analyze_performance_by_size(self, executions: List[Dict]) -> Dict:
        """Analyze performance metrics by order size."""
        
        small_orders = [e for e in executions if e['requested_quantity'] < 10]
        medium_orders = [e for e in executions if 10 <= e['requested_quantity'] < 100]
        large_orders = [e for e in executions if e['requested_quantity'] >= 100]
        
        def avg_slippage(orders):
            return np.mean([o['total_slippage_bps'] for o in orders]) if orders else 0
        
        return {
            'small_orders': {'count': len(small_orders), 'avg_slippage': avg_slippage(small_orders)},
            'medium_orders': {'count': len(medium_orders), 'avg_slippage': avg_slippage(medium_orders)},
            'large_orders': {'count': len(large_orders), 'avg_slippage': avg_slippage(large_orders)}
        }
    
    def _analyze_performance_by_volatility(self, executions: List[Dict]) -> Dict:
        """Analyze performance by market volatility."""
        
        low_vol = [e for e in executions if e['market_conditions']['volatility'] < 0.01]
        med_vol = [e for e in executions if 0.01 <= e['market_conditions']['volatility'] < 0.03]
        high_vol = [e for e in executions if e['market_conditions']['volatility'] >= 0.03]
        
        def avg_slippage(orders):
            return np.mean([o['total_slippage_bps'] for o in orders]) if orders else 0
        
        return {
            'low_volatility': {'count': len(low_vol), 'avg_slippage': avg_slippage(low_vol)},
            'medium_volatility': {'count': len(med_vol), 'avg_slippage': avg_slippage(med_vol)},
            'high_volatility': {'count': len(high_vol), 'avg_slippage': avg_slippage(high_vol)}
        }

# Testing and validation
async def test_smart_execution():
    """Test smart execution engine with various scenarios."""
    
    engine = SmartExecutionEngine()
    
    # Test data
    market_data = MarketMicrostructure(
        bid_price=50000.0,
        ask_price=50010.0,
        bid_size=10.0,
        ask_size=8.0,
        last_price=50005.0,
        volume_1min=100.0,
        volume_5min=400.0,
        volatility=0.015,
        tick_size=0.01,
        timestamp=datetime.utcnow()
    )
    
    # Test scenarios
    test_cases = [
        # Small urgent order
        TradingSignal(
            symbol='BTCUSDT',
            side='BUY',
            quantity=1.0,
            signal_strength=0.8,
            urgency=0.9
        ),
        # Large patient order
        TradingSignal(
            symbol='BTCUSDT',
            side='SELL',
            quantity=50.0,
            signal_strength=0.6,
            urgency=0.2,
            time_limit=timedelta(hours=1)
        ),
        # Medium order with high signal strength
        TradingSignal(
            symbol='BTCUSDT',
            side='BUY',
            quantity=10.0,
            signal_strength=0.9,
            urgency=0.5
        )
    ]
    
    for i, signal in enumerate(test_cases):
        print(f"\\nTest Case {i+1}: {signal.side} {signal.quantity} {signal.symbol}")
        print(f"Urgency: {signal.urgency}, Signal: {signal.signal_strength}")
        
        results = await engine.execute_signal(signal, market_data)
        
        total_filled = sum(r.fill_quantity for r in results)
        avg_price = sum(r.fill_price * r.fill_quantity for r in results) / total_filled
        avg_slippage = sum(r.slippage_bps for r in results) / len(results)
        
        print(f"Results: {total_filled:.3f} filled at {avg_price:.2f}")
        print(f"Average slippage: {avg_slippage:.2f} bps")
        print(f"Venues used: {[r.venue for r in results]}")
        
        # Validate results
        assert total_filled > 0, "Should have some fills"
        assert avg_slippage < 50, "Slippage should be reasonable"
        assert len(results) > 0, "Should have execution results"
    
    print("\\n✓ All smart execution tests passed")

# Usage example
async def implement_smart_execution():
    """Example of implementing smart execution in trading system."""
    
    # Initialize enhanced execution engine
    smart_engine = SmartExecutionEngine()
    performance_tracker = ExecutionPerformanceTracker()
    
    # Example trading signal
    signal = TradingSignal(
        symbol='BTCUSDT',
        side='BUY',
        quantity=25.0,
        signal_strength=0.75,
        urgency=0.4,
        max_participation_rate=0.15,
        time_limit=timedelta(minutes=45)
    )
    
    # Current market data (would come from data feed)
    market_data = MarketMicrostructure(
        bid_price=50000.0,
        ask_price=50015.0,
        bid_size=15.0,
        ask_size=12.0,
        last_price=50008.0,
        volume_1min=150.0,
        volume_5min=600.0,
        volatility=0.018,
        tick_size=0.01,
        timestamp=datetime.utcnow()
    )
    
    # Execute with smart routing
    execution_results = await smart_engine.execute_signal(signal, market_data)
    
    # Track performance
    await performance_tracker.record_execution(signal, execution_results, market_data)
    
    # Generate summary
    total_executed = sum(r.fill_quantity for r in execution_results)
    fill_rate = total_executed / signal.quantity
    avg_price = sum(r.fill_price * r.fill_quantity for r in execution_results) / total_executed
    
    print(f"Smart Execution Summary:")
    print(f"Requested: {signal.quantity} {signal.symbol}")
    print(f"Executed: {total_executed:.3f} ({fill_rate:.1%} fill rate)")
    print(f"Average Price: {avg_price:.2f}")
    print(f"Venues: {set(r.venue for r in execution_results)}")
    
    return execution_results
```
```

## 🛡️ Security and Type Safety (Week 4)

### Prompt 9: Implement Comprehensive Type Safety with Modern Python Standards
```
Add comprehensive type hints and domain-specific types using the latest Python typing features.

Create `types/trading_types.py`:

```python
from __future__ import annotations
from typing import (
    NewType, TypedDict, Literal, Optional, Union, Generic, TypeVar, 
    Protocol, runtime_checkable, Annotated, get_type_hints
)
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Domain-specific types with validation
Price = NewType('Price', Decimal)
Quantity = NewType('Quantity', Decimal)
Symbol = NewType('Symbol', str)
VenueID = NewType('VenueID', str)
OrderID = NewType('OrderID', str)
Percentage = NewType('Percentage', float)  # 0.0 to 1.0
BasisPoints = NewType('BasisPoints', float)  # 0.0 to 10000.0

# Literal types for constrained values
Side = Literal['BUY', 'SELL']
OrderStatus = Literal['PENDING', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED']
OrderType = Literal['MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT', 'TWAP', 'VWAP']
TimeInForce = Literal['GTC', 'IOC', 'FOK', 'GTD']

class StrategyType(Enum):
    """Enumeration of available trading strategies."""
    MOMENTUM = auto()
    MEAN_REVERSION = auto()
    ARBITRAGE = auto()
    MARKET_MAKING = auto()
    TREND_FOLLOWING = auto()

class RiskLevel(Enum):
    """Risk level classification."""
    CONSERVATIVE = auto()
    MODERATE = auto()
    AGGRESSIVE = auto()

# TypedDict for structured data with required/optional fields
class MarketDataDict(TypedDict, total=False):
    """Type definition for market data dictionary."""
    symbol: Symbol
    timestamp: datetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Quantity
    vwap: Optional[Price]
    trades_count: Optional[int]

class OrderDict(TypedDict):
    """Type definition for order dictionary."""
    order_id: OrderID
    symbol: Symbol
    side: Side
    order_type: OrderType
    quantity: Quantity
    price: Optional[Price]
    status: OrderStatus
    timestamp: datetime
    venue: VenueID
    time_in_force: TimeInForce

class PositionDict(TypedDict):
    """Type definition for position dictionary."""
    symbol: Symbol
    quantity: Quantity
    avg_entry_price: Price
    current_price: Price
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    side: Side
    timestamp: datetime

class RiskMetricsDict(TypedDict, total=False):
    """Type definition for risk metrics."""
    value_at_risk: Decimal
    expected_shortfall: Decimal
    sharpe_ratio: float
    max_drawdown: Percentage
    beta: float
    volatility: Percentage
    var_confidence: Percentage

# Generic types for containers
T = TypeVar('T')
PriceDataPoint = TypeVar('PriceDataPoint', bound=Union[Price, Decimal, float])

class TimeSeriesData(Generic[T]):
    """Generic time series data container."""
    
    def __init__(self, data: list[tuple[datetime, T]]):
        self.data = data
    
    def get_latest(self) -> tuple[datetime, T]:
        """Get the most recent data point."""
        return self.data[-1]
    
    def get_values(self) -> list[T]:
        """Get all values without timestamps."""
        return [value for _, value in self.data]
    
    def filter_by_time(self, start: datetime, end: datetime) -> Self:
        """Filter data by time range."""
        filtered = [(ts, val) for ts, val in self.data if start <= ts <= end]
        return self.__class__(filtered)

# Protocol definitions for structural typing
@runtime_checkable
class TradingStrategy(Protocol):
    """Protocol defining the interface for trading strategies."""
    
    def generate_signals(self, market_data: MarketDataDict) -> list[OrderDict]:
        """Generate trading signals from market data."""
        ...
    
    def calculate_position_size(self, signal_strength: float, 
                              available_capital: Decimal) -> Quantity:
        """Calculate appropriate position size."""
        ...
    
    def get_risk_parameters(self) -> RiskMetricsDict:
        """Get strategy risk parameters."""
        ...

@runtime_checkable
class RiskManager(Protocol):
    """Protocol for risk management systems."""
    
    def validate_order(self, order: OrderDict) -> bool:
        """Validate order against risk limits."""
        ...
    
    def calculate_portfolio_risk(self, positions: list[PositionDict]) -> RiskMetricsDict:
        """Calculate portfolio-level risk metrics."""
        ...
    
    def check_concentration_limits(self, symbol: Symbol, quantity: Quantity) -> bool:
        """Check if order violates concentration limits."""
        ...

@runtime_checkable
class ExecutionEngine(Protocol):
    """Protocol for order execution engines."""
    
    async def submit_order(self, order: OrderDict) -> OrderID:
        """Submit order for execution."""
        ...
    
    async def cancel_order(self, order_id: OrderID) -> bool:
        """Cancel existing order."""
        ...
    
    async def get_order_status(self, order_id: OrderID) -> OrderStatus:
        """Get current order status."""
        ...

# Advanced type annotations with validation
from pydantic import BaseModel, Field, validator, root_validator
from typing import Annotated

# Annotated types with runtime validation
ValidatedPrice = Annotated[Decimal, Field(gt=0, decimal_places=8)]
ValidatedQuantity = Annotated[Decimal, Field(gt=0, decimal_places=8)]
ValidatedPercentage = Annotated[float, Field(ge=0.0, le=1.0)]
ValidatedBasisPoints = Annotated[float, Field(ge=0.0, le=10000.0)]

class TypedOrder(BaseModel):
    """Typed order with runtime validation."""
    
    order_id: OrderID
    symbol: Symbol
    side: Side
    order_type: OrderType
    quantity: ValidatedQuantity
    price: Optional[ValidatedPrice] = None
    status: OrderStatus = 'PENDING'
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    venue: VenueID
    time_in_force: TimeInForce = 'GTC'
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate trading symbol format."""
        if not v or len(v) < 3:
            raise ValueError("Symbol must be at least 3 characters")
        if not v.replace('/', '').replace('-', '').isalnum():
            raise ValueError("Symbol contains invalid characters")
        return v
    
    @root_validator
    def validate_order_consistency(cls, values):
        """Validate order field consistency."""
        order_type = values.get('order_type')
        price = values.get('price')
        
        if order_type in ['LIMIT', 'STOP_LOSS', 'TAKE_PROFIT'] and price is None:
            raise ValueError(f"Order type {order_type} requires a price")
        
        if order_type == 'MARKET' and price is not None:
            raise ValueError("Market orders should not have a price")
        
        return values

class TypedPosition(BaseModel):
    """Typed position with validation."""
    
    symbol: Symbol
    quantity: ValidatedQuantity
    avg_entry_price: ValidatedPrice
    current_price: ValidatedPrice
    side: Side
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.side == 'BUY':
            return (self.current_price - self.avg_entry_price) * self.quantity
        else:
            return (self.avg_entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_percentage(self) -> float:
        """Calculate unrealized P&L as percentage."""
        cost_basis = self.avg_entry_price * self.quantity
        return float(self.unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0

class TypedRiskMetrics(BaseModel):
    """Typed risk metrics with validation."""
    
    portfolio_value: ValidatedPrice
    value_at_risk_95: ValidatedPrice
    value_at_risk_99: ValidatedPrice
    expected_shortfall: ValidatedPrice
    sharpe_ratio: float
    max_drawdown: ValidatedPercentage
    volatility_annualized: ValidatedPercentage
    beta: float = Field(ge=-3.0, le=3.0)
    correlation_to_market: float = Field(ge=-1.0, le=1.0)
    
    @validator('sharpe_ratio')
    def validate_sharpe_ratio(cls, v):
        """Validate Sharpe ratio is reasonable."""
        if abs(v) > 5.0:
            raise ValueError("Sharpe ratio seems unrealistic")
        return v

# Function signature types for callbacks and handlers
from typing import Callable, Awaitable

OrderCallback = Callable[[TypedOrder], Awaitable[None]]
MarketDataCallback = Callable[[MarketDataDict], Awaitable[None]]
RiskAlertCallback = Callable[[str, RiskLevel], Awaitable[None]]

# Type aliases for complex return types
StrategySignals = list[tuple[Symbol, Side, Quantity, float]]  # symbol, side, qty, confidence
PortfolioPositions = dict[Symbol, TypedPosition]
VenueOrderbook = dict[str, tuple[list[tuple[Price, Quantity]], list[tuple[Price, Quantity]]]]  # venue -> (bids, asks)

# Union types for flexible parameters
OrderInput = Union[OrderDict, TypedOrder]
PositionInput = Union[PositionDict, TypedPosition]
PriceInput = Union[Price, Decimal, float]
QuantityInput = Union[Quantity, Decimal, float]

# Type guards for runtime type checking
def is_valid_symbol(value: str) -> bool:
    """Type guard for valid trading symbols."""
    return (isinstance(value, str) and 
            len(value) >= 3 and 
            value.replace('/', '').replace('-', '').isalnum())

def is_valid_price(value: Union[Price, Decimal, float]) -> bool:
    """Type guard for valid prices."""
    try:
        price_val = Decimal(str(value))
        return price_val > 0
    except (ValueError, TypeError):
        return False

def is_valid_quantity(value: Union[Quantity, Decimal, float]) -> bool:
    """Type guard for valid quantities."""
    try:
        qty_val = Decimal(str(value))
        return qty_val > 0
    except (ValueError, TypeError):
        return False

# Type conversion utilities
def to_price(value: PriceInput) -> Price:
    """Convert input to validated Price type."""
    if not is_valid_price(value):
        raise ValueError(f"Invalid price: {value}")
    return Price(Decimal(str(value)))

def to_quantity(value: QuantityInput) -> Quantity:
    """Convert input to validated Quantity type."""
    if not is_valid_quantity(value):
        raise ValueError(f"Invalid quantity: {value}")
    return Quantity(Decimal(str(value)))

def to_symbol(value: str) -> Symbol:
    """Convert string to validated Symbol type."""
    if not is_valid_symbol(value):
        raise ValueError(f"Invalid symbol: {value}")
    return Symbol(value.upper())

# Advanced generic types for trading algorithms
AlgorithmConfig = TypeVar('AlgorithmConfig', bound=BaseModel)
AlgorithmResult = TypeVar('AlgorithmResult')

class TradingAlgorithm(Generic[AlgorithmConfig, AlgorithmResult], Protocol):
    """Generic protocol for trading algorithms."""
    
    config: AlgorithmConfig
    
    def run(self, market_data: TimeSeriesData[MarketDataDict]) -> AlgorithmResult:
        """Run the algorithm on market data."""
        ...
    
    def validate_config(self) -> bool:
        """Validate algorithm configuration."""
        ...

# Type-safe configuration classes
class MomentumStrategyConfig(BaseModel):
    """Configuration for momentum strategy."""
    
    lookback_period: int = Field(ge=1, le=252)
    momentum_threshold: ValidatedPercentage = 0.02
    position_size_pct: ValidatedPercentage = 0.1
    stop_loss_pct: ValidatedPercentage = 0.05
    take_profit_pct: ValidatedPercentage = 0.15
    max_positions: int = Field(ge=1, le=50)

class MeanReversionConfig(BaseModel):
    """Configuration for mean reversion strategy."""
    
    lookback_window: int = Field(ge=5, le=100)
    std_dev_threshold: float = Field(ge=1.0, le=5.0)
    holding_period: int = Field(ge=1, le=30)
    position_size_pct: ValidatedPercentage = 0.05
    max_drawdown_limit: ValidatedPercentage = 0.1

# Type-safe factory pattern
class StrategyFactory:
    """Type-safe factory for creating trading strategies."""
    
    @staticmethod
    def create_momentum_strategy(config: MomentumStrategyConfig) -> TradingStrategy:
        """Create momentum strategy with type safety."""
        from strategies.momentum_strategy import MomentumStrategy
        return MomentumStrategy(config)
    
    @staticmethod
    def create_mean_reversion_strategy(config: MeanReversionConfig) -> TradingStrategy:
        """Create mean reversion strategy with type safety."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy
        return MeanReversionStrategy(config)

# Type checking utilities for development
def check_types_at_runtime(func):
    """Decorator to check function argument types at runtime during development."""
    import functools
    import inspect
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints
        hints = get_type_hints(func)
        
        # Check positional arguments
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name, value in bound_args.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                if not isinstance(value, expected_type):
                    # For NewType, check the underlying type
                    if hasattr(expected_type, '__supertype__'):
                        if not isinstance(value, expected_type.__supertype__):
                            raise TypeError(f"Argument {param_name} expected {expected_type}, got {type(value)}")
                    else:
                        raise TypeError(f"Argument {param_name} expected {expected_type}, got {type(value)}")
        
        return func(*args, **kwargs)
    
    return wrapper

# Example usage of type-safe trading functions
@check_types_at_runtime
def calculate_position_value(position: TypedPosition, current_price: Price) -> Decimal:
    """Calculate position value with type safety."""
    return position.quantity * current_price

@check_types_at_runtime
def calculate_portfolio_risk(positions: list[TypedPosition], 
                           correlation_matrix: dict[tuple[Symbol, Symbol], float]) -> TypedRiskMetrics:
    """Calculate portfolio risk metrics with type safety."""
    # Implementation would go here
    total_value = sum(pos.market_value for pos in positions)
    
    return TypedRiskMetrics(
        portfolio_value=Price(total_value),
        value_at_risk_95=Price(total_value * Decimal('0.05')),
        value_at_risk_99=Price(total_value * Decimal('0.01')),
        expected_shortfall=Price(total_value * Decimal('0.025')),
        sharpe_ratio=1.5,
        max_drawdown=0.08,
        volatility_annualized=0.15,
        beta=1.2,
        correlation_to_market=0.8
    )

# Integration with existing codebase
def upgrade_existing_functions_with_types():
    """Example of upgrading existing functions with proper types."""
    
    # Old function (no types)
    def old_create_order(symbol, side, quantity, price=None):
        return {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.utcnow()
        }
    
    # New function (fully typed)
    def create_typed_order(symbol: Symbol, side: Side, quantity: Quantity, 
                          price: Optional[Price] = None, 
                          venue: VenueID = VenueID('DEFAULT')) -> TypedOrder:
        """Create a new order with full type safety."""
        return TypedOrder(
            order_id=OrderID(f"ORD_{datetime.utcnow().timestamp()}"),
            symbol=symbol,
            side=side,
            order_type='LIMIT' if price else 'MARKET',
            quantity=quantity,
            price=price,
            venue=venue
        )
    
    return create_typed_order

# Performance impact measurement
def measure_type_checking_overhead():
    """Measure performance impact of runtime type checking."""
    import time
    import random
    
    # Create test data
    test_positions = [
        TypedPosition(
            symbol=Symbol(f"SYM{i}"),
            quantity=Quantity(Decimal(random.uniform(1, 100))),
            avg_entry_price=Price(Decimal(random.uniform(50, 200))),
            current_price=Price(Decimal(random.uniform(50, 200))),
            side='BUY'
        )
        for i in range(1000)
    ]
    
    # Test with type checking
    start = time.perf_counter()
    for pos in test_positions:
        value = calculate_position_value(pos, pos.current_price)
    typed_time = time.perf_counter() - start
    
    # Test without type checking (remove decorator)
    def calculate_position_value_untyped(position, current_price):
        return position.quantity * current_price
    
    start = time.perf_counter()
    for pos in test_positions:
        value = calculate_position_value_untyped(pos, pos.current_price)
    untyped_time = time.perf_counter() - start
    
    overhead = (typed_time - untyped_time) / untyped_time * 100
    print(f"Type checking overhead: {overhead:.2f}%")
    
    return overhead
```

Create `mypy.ini` configuration:
```ini
[mypy]
python_version = 3.11
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
implicit_reexport = False
strict_equality = True

# Module-specific overrides
[mypy-numpy.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-ta.*]
ignore_missing_imports = True

[mypy-ccxt.*]
ignore_missing_imports = True

# Trading module settings
[mypy-src.trading.*]
strict = True
disallow_any_unimported = True
disallow_any_expr = True
disallow_any_decorated = True
disallow_any_explicit = True
disallow_subclass_any = True
```

Update key trading modules with comprehensive types:
```python
# Example: Updated strategies/base_strategy.py
from abc import ABC, abstractmethod
from types.trading_types import (
    TradingStrategy, MarketDataDict, OrderDict, TypedPosition,
    StrategySignals, Symbol, Quantity, RiskMetricsDict
)

class BaseStrategy(ABC):
    """Base class for all trading strategies with full type safety."""
    
    def __init__(self, name: str, risk_level: RiskLevel) -> None:
        self.name = name
        self.risk_level = risk_level
        self._positions: dict[Symbol, TypedPosition] = {}
    
    @abstractmethod
    def generate_signals(self, market_data: MarketDataDict) -> list[OrderDict]:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal_strength: float, 
                              available_capital: Decimal) -> Quantity:
        """Calculate appropriate position size."""
        pass
    
    def get_positions(self) -> dict[Symbol, TypedPosition]:
        """Get current positions."""
        return self._positions.copy()
    
    def update_position(self, position: TypedPosition) -> None:
        """Update position in strategy."""
        self._positions[position.symbol] = position
    
    def get_risk_parameters(self) -> RiskMetricsDict:
        """Get strategy risk parameters."""
        return {
            'max_drawdown': 0.05,
            'volatility': 0.15,
            'var_confidence': 0.95
        }

# Example: Updated execution_engine.py with types
from types.trading_types import (
    ExecutionEngine, OrderDict, OrderID, OrderStatus, TypedOrder
)

class TypedExecutionEngine(ExecutionEngine):
    """Execution engine with comprehensive type safety."""
    
    def __init__(self) -> None:
        self._orders: dict[OrderID, TypedOrder] = {}
        self.logger = logging.getLogger(__name__)
    
    async def submit_order(self, order: OrderDict) -> OrderID:
        """Submit order for execution."""
        typed_order = TypedOrder(**order)
        self._orders[typed_order.order_id] = typed_order
        
        # Submit to exchange (implementation specific)
        success = await self._submit_to_exchange(typed_order)
        
        if success:
            typed_order.status = 'PENDING'
        else:
            typed_order.status = 'REJECTED'
        
        return typed_order.order_id
    
    async def cancel_order(self, order_id: OrderID) -> bool:
        """Cancel existing order."""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        if order.status in ['FILLED', 'CANCELLED']:
            return False
        
        # Cancel at exchange
        success = await self._cancel_at_exchange(order_id)
        if success:
            order.status = 'CANCELLED'
        
        return success
    
    async def get_order_status(self, order_id: OrderID) -> OrderStatus:
        """Get current order status."""
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")
        
        return self._orders[order_id].status
    
    async def _submit_to_exchange(self, order: TypedOrder) -> bool:
        """Submit order to exchange (placeholder)."""
        # Implementation would integrate with actual exchange API
        return True
    
    async def _cancel_at_exchange(self, order_id: OrderID) -> bool:
        """Cancel order at exchange (placeholder)."""
        # Implementation would integrate with actual exchange API
        return True
```

Test type safety implementation:
```python
async def test_type_safety():
    """Test comprehensive type safety implementation."""
    
    # Test type creation and validation
    try:
        # Valid types
        symbol = to_symbol("BTCUSDT")
        price = to_price(50000.50)
        quantity = to_quantity(1.5)
        
        # Create typed order
        order = TypedOrder(
            order_id=OrderID("TEST_001"),
            symbol=symbol,
            side='BUY',
            order_type='LIMIT',
            quantity=quantity,
            price=price,
            venue=VenueID('BINANCE')
        )
        
        print(f"✓ Created order: {order}")
        
        # Test validation failures
        try:
            invalid_order = TypedOrder(
                order_id=OrderID("TEST_002"),
                symbol=Symbol(""),  # Invalid empty symbol
                side='BUY',
                order_type='LIMIT',
                quantity=quantity,
                price=price,
                venue=VenueID('BINANCE')
            )
            assert False, "Should have failed validation"
        except ValueError:
            print("✓ Validation correctly rejected invalid symbol")
        
        # Test type checking
        try:
            result = calculate_position_value(
                TypedPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=price,
                    current_price=price,
                    side='BUY'
                ),
                price
            )
            print(f"✓ Type-safe calculation result: {result}")
        except TypeError as e:
            print(f"✗ Type checking failed: {e}")
        
        print("✓ All type safety tests passed")
        
    except Exception as e:
        print(f"✗ Type safety test failed: {e}")
        raise

# Performance measurement
def benchmark_type_safety():
    """Benchmark the performance impact of type safety."""
    
    overhead = measure_type_checking_overhead()
    print(f"Type checking overhead: {overhead:.2f}%")
    
    # Acceptable overhead threshold
    assert overhead < 10, f"Type checking overhead too high: {overhead:.2f}%"
    
    return overhead
```

Run type checking:
```bash
# Type check entire codebase
mypy src/trading/ --strict

# Check specific modules
mypy src/trading/strategies/ --strict
mypy src/trading/execution/ --strict
mypy src/trading/risk/ --strict

# Generate type coverage report
mypy src/trading/ --strict --html-report mypy_report/
```

## Enhanced Testing and Documentation (Month 3)

### Prompt 15: Implement Production-Ready Deployment with Modern DevOps
```
Create a comprehensive deployment pipeline with monitoring, logging, and observability.

Create `deployment/docker/Dockerfile`:

```dockerfile
# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading trading

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Set permissions
RUN chown -R trading:trading /app

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R trading:trading /app/logs /app/data

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose monitoring port
EXPOSE 8080

# Entry point
CMD ["python", "-m", "src.main"]
```

Create `deployment/docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  trading-bot:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile
      target: production
    environment:
      - DATABASE_URL=postgresql://trading_user:${DB_PASSWORD}@postgres:5432/trading_db
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - ENVIRONMENT=production
      - SENTRY_DSN=${SENTRY_DSN}
      - PROMETHEUS_PORT=8080
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8080:8080"  # Monitoring/health check port
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./config/production.yaml:/app/config/production.yaml:ro
    networks:
      - trading-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user -d trading_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G

  redis:
    image: redis:7-alpine
    command: >
      redis-server 
      --appendonly yes 
      --maxmemory 1gb 
      --maxmemory-policy allkeys-lru
      --tcp-keepalive 60
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - trading-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    networks:
      - trading-network

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - trading-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  trading-network:
    driver: bridge
```

Create comprehensive monitoring system:

```python
# monitoring/observability.py
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
import aiohttp
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, start_http_server,
    CollectorRegistry, generate_latest
)
import opentelemetry
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    prometheus_port: int = 8080
    metrics_interval: int = 30  # seconds
    jaeger_endpoint: str = "http://jaeger:14268/api/traces"
    sentry_dsn: Optional[str] = None
    service_name: str = "trading-bot"
    environment: str = "production"

class TradingMetrics:
    """Comprehensive metrics collection for trading bot."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        self._setup_opentelemetry()
        self._setup_sentry()
        self.logger = logging.getLogger(__name__)
        
        # System metrics
        self._last_metrics_time = time.time()
        self._system_metrics_task: Optional[asyncio.Task] = None
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        
        # Trading-specific metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders',
            ['side', 'order_type', 'venue', 'status'],
            registry=self.registry
        )
        
        self.order_latency = Histogram(
            'trading_order_latency_seconds',
            'Order execution latency',
            ['venue', 'order_type'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.slippage = Histogram(
            'trading_slippage_bps',
            'Trading slippage in basis points',
            ['symbol', 'side'],
            buckets=[0, 1, 5, 10, 25, 50, 100, 500],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'trading_portfolio_value_usd',
            'Total portfolio value in USD',
            registry=self.registry
        )
        
        self.open_positions = Gauge(
            'trading_open_positions',
            'Number of open positions',
            ['symbol'],
            registry=self.registry
        )
        
        self.pnl_realized = Counter(
            'trading_pnl_realized_usd',
            'Realized P&L in USD',
            ['symbol', 'strategy'],
            registry=self.registry
        )
        
        self.pnl_unrealized = Gauge(
            'trading_pnl_unrealized_usd',
            'Unrealized P&L in USD',
            ['symbol', 'strategy'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            ['mountpoint'],
            registry=self.registry
        )
        
        # Application metrics
        self.active_strategies = Gauge(
            'trading_active_strategies',
            'Number of active trading strategies',
            registry=self.registry
        )
        
        self.market_data_latency = Histogram(
            'trading_market_data_latency_seconds',
            'Market data feed latency',
            ['source', 'symbol'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.errors_total = Counter(
            'trading_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'trading_cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.database_operations = Histogram(
            'trading_database_operations_seconds',
            'Database operation duration',
            ['operation', 'table'],
            registry=self.registry
        )
    
    def _setup_opentelemetry(self):
        """Initialize OpenTelemetry tracing and metrics."""
        
        # Resource identification
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": "1.0.0",
            "deployment.environment": self.config.environment
        })
        
        # Tracing setup
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        
        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Metrics setup
        prometheus_reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader]
        ))
        
        self.meter = metrics.get_meter(__name__)
    
    def _setup_sentry(self):
        """Initialize Sentry error tracking."""
        
        if self.config.sentry_dsn:
            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )
            
            sentry_sdk.init(
                dsn=self.config.sentry_dsn,
                environment=self.config.environment,
                integrations=[
                    AsyncioIntegration(auto_enable_aiohttp_server=True),
                    sentry_logging,
                ],
                traces_sample_rate=0.1,
                attach_stacktrace=True,
                send_default_pii=False
            )
    
    async def start_metrics_server(self):
        """Start Prometheus metrics server."""
        
        start_http_server(self.config.prometheus_port, registry=self.registry)
        self.logger.info(f"Metrics server started on port {self.config.prometheus_port}")
        
        # Start system metrics collection
        self._system_metrics_task = asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_system_metrics(self):
        """Continuously collect system metrics."""
        
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                
                # Disk metrics
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        self.disk_usage.labels(mountpoint=partition.mountpoint).set(usage.used)
                    except PermissionError:
                        pass
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.config.metrics_interval)
    
    def record_order(self, order_data: Dict[str, Any], execution_time: float):
        """Record order metrics."""
        
        self.orders_total.labels(
            side=order_data.get('side', 'unknown'),
            order_type=order_data.get('order_type', 'unknown'),
            venue=order_data.get('venue', 'unknown'),
            status=order_data.get('status', 'unknown')
        ).inc()
        
        self.order_latency.labels(
            venue=order_data.get('venue', 'unknown'),
            order_type=order_data.get('order_type', 'unknown')
        ).observe(execution_time)
    
    def record_slippage(self, symbol: str, side: str, slippage_bps: float):
        """Record slippage metrics."""
        
        self.slippage.labels(symbol=symbol, side=side).observe(slippage_bps)
    
    def update_portfolio_metrics(self, portfolio_value: float, positions: Dict[str, Any]):
        """Update portfolio-related metrics."""
        
        self.portfolio_value.set(portfolio_value)
        
        # Clear existing position metrics
        self.open_positions.clear()
        
        # Update position metrics
        for symbol, position in positions.items():
            self.open_positions.labels(symbol=symbol).set(position.get('quantity', 0))
            
            # Update P&L metrics
            if 'unrealized_pnl' in position:
                self.pnl_unrealized.labels(
                    symbol=symbol,
                    strategy=position.get('strategy', 'unknown')
                ).set(position['unrealized_pnl'])
    
    def record_error(self, component: str, error_type: str):
        """Record error occurrence."""
        
        self.errors_total.labels(component=component, error_type=error_type).inc()
    
    def record_market_data_latency(self, source: str, symbol: str, latency: float):
        """Record market data latency."""
        
        self.market_data_latency.labels(source=source, symbol=symbol).observe(latency)
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation."""
        
        self.cache_operations.labels(operation=operation, result=result).inc()
    
    def record_database_operation(self, operation: str, table: str, duration: float):
        """Record database operation timing."""
        
        self.database_operations.labels(operation=operation, table=table).observe(duration)
    
    def create_span(self, name: str, **attributes):
        """Create a tracing span."""
        
        span = self.tracer.start_span(name)
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        return span
    
    async def shutdown(self):
        """Shutdown metrics collection."""
        
        if self._system_metrics_task:
            self._system_metrics_task.cancel()
            try:
                await self._system_metrics_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Metrics collection shutdown complete")

# Health check endpoint
from aiohttp import web, web_response
import json

class HealthCheckServer:
    """Health check and metrics endpoint server."""
    
    def __init__(self, metrics: TradingMetrics, trading_system):
        self.metrics = metrics
        self.trading_system = trading_system
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/ready', self.readiness_check)
        self.app.router.add_get('/metrics', self.prometheus_metrics)
        self.app.router.add_get('/status', self.detailed_status)
    
    async def health_check(self, request):
        """Basic health check endpoint."""
        
        try:
            # Basic system checks
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            if cpu_usage > 95 or memory.percent > 95:
                return web_response.json_response(
                    {'status': 'unhealthy', 'reason': 'high_resource_usage'},
                    status=503
                )
            
            return web_response.json_response({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            })
            
        except Exception as e:
            return web_response.json_response(
                {'status': 'error', 'error': str(e)},
                status=500
            )
    
    async def readiness_check(self, request):
        """Readiness check for Kubernetes."""
        
        try:
            # Check if trading system is ready
            if not hasattr(self.trading_system, 'is_ready') or not self.trading_system.is_ready():
                return web_response.json_response(
                    {'status': 'not_ready', 'reason': 'trading_system_not_ready'},
                    status=503
                )
            
            # Check database connectivity
            if hasattr(self.trading_system, 'database_manager'):
                db_status = await self._check_database()
                if not db_status:
                    return web_response.json_response(
                        {'status': 'not_ready', 'reason': 'database_unavailable'},
                        status=503
                    )
            
            # Check Redis connectivity
            if hasattr(self.trading_system, 'cache_manager'):
                cache_status = await self._check_cache()
                if not cache_status:
                    return web_response.json_response(
                        {'status': 'not_ready', 'reason': 'cache_unavailable'},
                        status=503
                    )
            
            return web_response.json_response({
                'status': 'ready',
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            return web_response.json_response(
                {'status': 'error', 'error': str(e)},
                status=500
            )
    
    async def prometheus_metrics(self, request):
        """Prometheus metrics endpoint."""
        
        try:
            metrics_output = generate_latest(self.metrics.registry)
            return web.Response(
                text=metrics_output.decode('utf-8'),
                content_type='text/plain; version=0.0.4; charset=utf-8'
            )
        except Exception as e:
            return web_response.json_response(
                {'error': f'Failed to generate metrics: {str(e)}'},
                status=500
            )
    
    async def detailed_status(self, request):
        """Detailed system status endpoint."""
        
        try:
            status = {
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': {
                        partition.mountpoint: psutil.disk_usage(partition.mountpoint).percent
                        for partition in psutil.disk_partitions()
                        if partition.mountpoint in ['/', '/tmp']
                    }
                },
                'trading': {
                    'active_strategies': getattr(self.trading_system, 'active_strategies_count', 0),
                    'open_positions': getattr(self.trading_system, 'open_positions_count', 0),
                    'last_trade_time': getattr(self.trading_system, 'last_trade_time', None),
                },
                'connectivity': {
                    'database': await self._check_database(),
                    'cache': await self._check_cache(),
                    'market_data': await self._check_market_data(),
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return web_response.json_response(status)
            
        except Exception as e:
            return web_response.json_response(
                {'error': f'Failed to get status: {str(e)}'},
                status=500
            )
    
    async def _check_database(self) -> bool:
        """Check database connectivity."""
        
        try:
            if hasattr(self.trading_system, 'database_manager'):
                # Perform a simple query to test connectivity
                async with self.trading_system.database_manager.acquire_connection() as conn:
                    await conn.fetchval('SELECT 1')
                return True
        except Exception:
            pass
        
        return False
    
    async def _check_cache(self) -> bool:
        """Check Redis cache connectivity."""
        
        try:
            if hasattr(self.trading_system, 'cache_manager'):
                # Test Redis connectivity
                await self.trading_system.cache_manager.redis_client.ping()
                return True
        except Exception:
            pass
        
        return False
    
    async def _check_market_data(self) -> bool:
        """Check market data feed connectivity."""
        
        try:
            if hasattr(self.trading_system, 'market_data_manager'):
                # Check if market data is current (within last 60 seconds)
                last_update = getattr(self.trading_system.market_data_manager, 'last_update_time', None)
                if last_update and (datetime.utcnow() - last_update).total_seconds() < 60:
                    return True
        except Exception:
            pass
        
        return False
    
    async def start_server(self, port: int = 8080):
        """Start the health check server."""
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        return runner

# Integration with main trading system
class MonitoredTradingSystem:
    """Trading system with comprehensive monitoring."""
    
    def __init__(self, config: MetricsConfig):
        self.metrics = TradingMetrics(config)
        self.health_server = HealthCheckServer(self.metrics, self)
        self.logger = logging.getLogger(__name__)
        self._ready = False
        self.active_strategies_count = 0
        self.open_positions_count = 0
        self.last_trade_time: Optional[datetime] = None
    
    async def start(self):
        """Start the monitored trading system."""
        
        # Start metrics collection
        await self.metrics.start_metrics_server()
        
        # Start health check server
        await self.health_server.start_server()
        
        # Initialize trading components
        await self._initialize_trading_components()
        
        self._ready = True
        self.logger.info("Monitored trading system started successfully")
    
    async def _initialize_trading_components(self):
        """Initialize trading system components."""
        
        # Initialize database, cache, strategies, etc.
        # This would contain your actual trading system initialization
        pass
    
    def is_ready(self) -> bool:
        """Check if system is ready."""
        return self._ready
    
    async def execute_trade_with_monitoring(self, trade_data: Dict[str, Any]):
        """Execute trade with comprehensive monitoring."""
        
        start_time = time.time()
        
        # Create tracing span
        with self.metrics.create_span("trade_execution") as span:
            span.set_attribute("symbol", trade_data.get('symbol', 'unknown'))
            span.set_attribute("side", trade_data.get('side', 'unknown'))
            span.set_attribute("quantity", trade_data.get('quantity', 0))
            
            try:
                # Execute trade (your implementation)
                result = await self._execute_trade_impl(trade_data)
                
                # Record success metrics
                execution_time = time.time() - start_time
                self.metrics.record_order(trade_data, execution_time)
                
                # Record slippage if available
                if 'slippage_bps' in result:
                    self.metrics.record_slippage(
                        trade_data['symbol'],
                        trade_data['side'],
                        result['slippage_bps']
                    )
                
                span.set_attribute("execution_time", execution_time)
                span.set_attribute("status", "success")
                
                # Update last trade time
                self.last_trade_time = datetime.utcnow()
                
                return result
                
            except Exception as e:
                # Record error metrics
                self.metrics.record_error("trade_execution", type(e).__name__)
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                
                # Re-raise for handling upstream
                raise
    
    async def _execute_trade_impl(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Actual trade execution implementation."""
        
        # Placeholder for actual trade execution
        await asyncio.sleep(0.1)  # Simulate execution time
        
        return {
            'order_id': f"ORD_{int(time.time())}",
            'status': 'FILLED',
            'fill_price': 50000.0,
            'slippage_bps': 2.5
        }
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        
        self.logger.info("Shutting down monitored trading system...")
        
        # Shutdown metrics collection
        await self.metrics.shutdown()
        
        self._ready = False
        self.logger.info("Shutdown complete")

# Usage example
async def run_production_system():
    """Run the production trading system with full monitoring."""
    
    config = MetricsConfig(
        prometheus_port=8080,
        jaeger_endpoint="http://jaeger:14268/api/traces",
        sentry_dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("ENVIRONMENT", "production")
    )
    
    system = MonitoredTradingSystem(config)
    
    try:
        await system.start()
        
        # Keep system running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("Received shutdown signal")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(run_production_system())
```

Create monitoring configuration files:

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8080']
    scrape_interval: 10s
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

Create deployment scripts:

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
COMPOSE_FILE="deployment/docker-compose.prod.yml"

echo "Deploying trading bot to $ENVIRONMENT environment..."

# Load environment variables
if [ -f ".env.$ENVIRONMENT" ]; then
    source ".env.$ENVIRONMENT"
fi

# Build and deploy
docker-compose -f $COMPOSE_FILE down
docker-compose -f $COMPOSE_FILE build --no-cache
docker-compose -f $COMPOSE_FILE up -d

# Health check
echo "Waiting for services to start..."
sleep 30

# Check health endpoints
curl -f http://localhost:8080/health || {
    echo "Health check failed!"
    docker-compose -f $COMPOSE_FILE logs trading-bot
    exit 1
}

echo "Deployment successful!"

# Run smoke tests
echo "Running smoke tests..."
python scripts/smoke_tests.py

echo "All tests passed. Deployment complete!"
```

```python
# scripts/smoke_tests.py
import asyncio
import aiohttp
import sys
import time

async def run_smoke_tests():
    """Run smoke tests against deployed system."""
    
    base_url = "http://localhost:8080"
    
    tests = [
        ("Health Check", f"{base_url}/health"),
        ("Readiness Check", f"{base_url}/ready"),
        ("Metrics Endpoint", f"{base_url}/metrics"),
        ("Status Endpoint", f"{base_url}/status"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for test_name, url in tests:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        print(f"✓ {test_name}: PASSED")
                    else:
                        print(f"✗ {test_name}: FAILED (status {response.status})")
                        return False
            except Exception as e:
                print(f"✗ {test_name}: FAILED ({str(e)})")
                return False
    
    print("All smoke tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(run_smoke_tests())
    sys.exit(0 if success else 1)
```

## Verification and Testing Strategy

### Enhanced Verification Commands

```bash
# Type checking with comprehensive coverage
mypy src/ --strict --show-error-codes --show-column-numbers

# Security scanning
bandit -r src/ -f json -o security_report.json
safety check --json --output safety_report.json

# Performance benchmarking
python -m pytest tests/benchmarks/ -v --benchmark-only

# Memory profiling
python -m memory_profiler src/main.py

# Load testing
python tests/load/load_test.py --users 100 --duration 300

# Integration testing with Docker
docker-compose -f tests/docker-compose.test.yml up --abort-on-container-exit

# Monitoring validation
python scripts/validate_monitoring.py

# Database migration testing
python scripts/test_migrations.py

# Cache performance validation
python scripts/validate_cache_performance.py
```

This enhanced trading bot optimization guide incorporates the latest 2025 best practices for:

1. **Modern async programming patterns** with proper error handling and resource management
2. **Advanced type safety** using the latest Python typing features
3. **High-performance computing** with Numba JIT compilation
4. **Intelligent order execution** with market microstructure awareness
5. **Production-grade deployment** with comprehensive monitoring and observability
6. **Advanced caching strategies** using Redis with intelligent invalidation
7. **Memory optimization** techniques for large-scale data processing
8. **Comprehensive security measures** including input validation and secrets management

The guide now provides a complete roadmap for transforming a basic trading bot into a production-ready, institutional-grade system that can handle high-frequency trading with proper risk management, monitoring, and scalability.
