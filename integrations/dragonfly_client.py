# =======================================================================

#!/usr/bin/env python3
"""
DragonflyDB Redis Client - Production Implementation
Handles Redis-compatible operations with DragonflyDB cloud service
"""

import os
import redis
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urlparse
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DragonflyClient:
    """Production-ready DragonflyDB client with full Redis compatibility"""

    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url or os.getenv('DRAGONFLY_REDIS_URL')
        if not self.connection_url:
            raise ValueError("DRAGONFLY_REDIS_URL environment variable is required")

        self.client = self._create_connection()
        self._verify_connection()

    def _create_connection(self) -> redis.Redis:
        """Create Redis connection with SSL support for DragonflyDB Cloud"""
        try:
            parsed_url = urlparse(self.connection_url)

            # Configure SSL for secure connections
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            client = redis.Redis.from_url(
                self.connection_url,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
                ssl_cert_reqs=ssl.CERT_NONE,
                ssl_ca_certs=None,
                ssl_check_hostname=False
            )

            logger.info(f"Connected to DragonflyDB at {parsed_url.hostname}:{parsed_url.port}")
            return client

        except Exception as e:
            logger.error(f"Failed to create DragonflyDB connection: {e}")
            raise

    def _verify_connection(self) -> None:
        """Verify connection is working"""
        try:
            response = self.client.ping()
            if response:
                logger.info("DragonflyDB connection verified successfully")
            else:
                raise ConnectionError("Failed to ping DragonflyDB")
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            raise

    # String Operations
    def set(self, key: str, value: Any, ex: Optional[int] = None, px: Optional[int] = None,
           nx: bool = False, xx: bool = False) -> bool:
        """Set a key-value pair with optional expiration"""
        try:
            return self.client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            raise

    def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        try:
            return self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Failed to delete keys {keys}: {e}")
            raise

    def exists(self, *keys: str) -> int:
        """Check if keys exist"""
        try:
            return self.client.exists(*keys)
        except Exception as e:
            logger.error(f"Failed to check existence of keys {keys}: {e}")
            raise

    def expire(self, key: str, time: int) -> bool:
        """Set key expiration time in seconds"""
        try:
            return self.client.expire(key, time)
        except Exception as e:
            logger.error(f"Failed to set expiration for key {key}: {e}")
            raise

    def ttl(self, key: str) -> int:
        """Get time to live for key"""
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            raise

    # Hash Operations
    def hset(self, name: str, key: str = None, value: Any = None, mapping: Dict = None) -> int:
        """Set hash field"""
        try:
            if mapping:
                return self.client.hset(name, mapping=mapping)
            else:
                return self.client.hset(name, key, value)
        except Exception as e:
            logger.error(f"Failed to set hash {name}: {e}")
            raise

    def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value"""
        try:
            return self.client.hget(name, key)
        except Exception as e:
            logger.error(f"Failed to get hash field {name}:{key}: {e}")
            raise

    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields and values"""
        try:
            return self.client.hgetall(name)
        except Exception as e:
            logger.error(f"Failed to get all hash fields for {name}: {e}")
            raise

    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        try:
            return self.client.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Failed to delete hash fields {name}:{keys}: {e}")
            raise

    def hexists(self, name: str, key: str) -> bool:
        """Check if hash field exists"""
        try:
            return self.client.hexists(name, key)
        except Exception as e:
            logger.error(f"Failed to check hash field existence {name}:{key}: {e}")
            raise

    # List Operations
    def lpush(self, name: str, *values: Any) -> int:
        """Push values to left of list"""
        try:
            return self.client.lpush(name, *values)
        except Exception as e:
            logger.error(f"Failed to lpush to list {name}: {e}")
            raise

    def rpush(self, name: str, *values: Any) -> int:
        """Push values to right of list"""
        try:
            return self.client.rpush(name, *values)
        except Exception as e:
            logger.error(f"Failed to rpush to list {name}: {e}")
            raise

    def lpop(self, name: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        """Pop value from left of list"""
        try:
            return self.client.lpop(name, count)
        except Exception as e:
            logger.error(f"Failed to lpop from list {name}: {e}")
            raise

    def rpop(self, name: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        """Pop value from right of list"""
        try:
            return self.client.rpop(name, count)
        except Exception as e:
            logger.error(f"Failed to rpop from list {name}: {e}")
            raise

    def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Get range of list elements"""
        try:
            return self.client.lrange(name, start, end)
        except Exception as e:
            logger.error(f"Failed to get range from list {name}: {e}")
            raise

    def llen(self, name: str) -> int:
        """Get list length"""
        try:
            return self.client.llen(name)
        except Exception as e:
            logger.error(f"Failed to get length of list {name}: {e}")
            raise

    # Set Operations
    def sadd(self, name: str, *values: Any) -> int:
        """Add values to set"""
        try:
            return self.client.sadd(name, *values)
        except Exception as e:
            logger.error(f"Failed to add to set {name}: {e}")
            raise

    def srem(self, name: str, *values: Any) -> int:
        """Remove values from set"""
        try:
            return self.client.srem(name, *values)
        except Exception as e:
            logger.error(f"Failed to remove from set {name}: {e}")
            raise

    def smembers(self, name: str) -> set:
        """Get all set members"""
        try:
            return self.client.smembers(name)
        except Exception as e:
            logger.error(f"Failed to get set members {name}: {e}")
            raise

    def sismember(self, name: str, value: Any) -> bool:
        """Check if value is in set"""
        try:
            return self.client.sismember(name, value)
        except Exception as e:
            logger.error(f"Failed to check set membership {name}: {e}")
            raise

    def scard(self, name: str) -> int:
        """Get set cardinality"""
        try:
            return self.client.scard(name)
        except Exception as e:
            logger.error(f"Failed to get set cardinality {name}: {e}")
            raise

    # Sorted Set Operations
    def zadd(self, name: str, mapping: Dict[Any, float], nx: bool = False, xx: bool = False,
            ch: bool = False, incr: bool = False) -> Union[int, float]:
        """Add members to sorted set"""
        try:
            return self.client.zadd(name, mapping, nx=nx, xx=xx, ch=ch, incr=incr)
        except Exception as e:
            logger.error(f"Failed to add to sorted set {name}: {e}")
            raise

    def zrange(self, name: str, start: int, end: int, desc: bool = False,
              withscores: bool = False, score_cast_func=float) -> List:
        """Get range from sorted set"""
        try:
            return self.client.zrange(name, start, end, desc=desc, withscores=withscores,
                                    score_cast_func=score_cast_func)
        except Exception as e:
            logger.error(f"Failed to get range from sorted set {name}: {e}")
            raise

    def zrem(self, name: str, *values: Any) -> int:
        """Remove members from sorted set"""
        try:
            return self.client.zrem(name, *values)
        except Exception as e:
            logger.error(f"Failed to remove from sorted set {name}: {e}")
            raise

    def zcard(self, name: str) -> int:
        """Get sorted set cardinality"""
        try:
            return self.client.zcard(name)
        except Exception as e:
            logger.error(f"Failed to get sorted set cardinality {name}: {e}")
            raise

    def zscore(self, name: str, value: Any) -> Optional[float]:
        """Get score of member in sorted set"""
        try:
            return self.client.zscore(name, value)
        except Exception as e:
            logger.error(f"Failed to get score from sorted set {name}: {e}")
            raise

    # JSON Operations (DragonflyDB supports JSON)
    def json_set(self, key: str, path: str, obj: Any) -> bool:
        """Set JSON object"""
        try:
            return self.client.execute_command('JSON.SET', key, path, json.dumps(obj))
        except Exception as e:
            logger.error(f"Failed to set JSON {key}: {e}")
            raise

    def json_get(self, key: str, path: str = '.') -> Any:
        """Get JSON object"""
        try:
            result = self.client.execute_command('JSON.GET', key, path)
            return json.loads(result) if result else None
        except Exception as e:
            logger.error(f"Failed to get JSON {key}: {e}")
            raise

    def json_del(self, key: str, path: str = '.') -> int:
        """Delete JSON path"""
        try:
            return self.client.execute_command('JSON.DEL', key, path)
        except Exception as e:
            logger.error(f"Failed to delete JSON {key}: {e}")
            raise

    # Advanced Operations
    def pipeline(self):
        """Create pipeline for batch operations"""
        return self.client.pipeline()

    def transaction(self):
        """Create transaction"""
        return self.client.pipeline(transaction=True)

    def pubsub(self):
        """Create pub/sub connection"""
        return self.client.pubsub()

    def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        try:
            return self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            raise

    def keys(self, pattern: str = '*') -> List[str]:
        """Get keys matching pattern"""
        try:
            return self.client.keys(pattern)
        except Exception as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            raise

    def flushdb(self) -> bool:
        """Flush current database"""
        try:
            return self.client.flushdb()
        except Exception as e:
            logger.error(f"Failed to flush database: {e}")
            raise

    def flushall(self) -> bool:
        """Flush all databases"""
        try:
            return self.client.flushall()
        except Exception as e:
            logger.error(f"Failed to flush all databases: {e}")
            raise

    def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get server info"""
        try:
            return self.client.info(section)
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            raise

    def ping(self) -> bool:
        """Ping server"""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Failed to ping server: {e}")
            raise

    def close(self) -> None:
        """Close connection"""
        try:
            self.client.close()
            logger.info("DragonflyDB connection closed")
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
# =======================================================================


# =======================================================================
