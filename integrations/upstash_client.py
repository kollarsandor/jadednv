# =======================================================================

#!/usr/bin/env python3
"""
Upstash Redis and Search Client - Production Implementation
Handles both Redis and Vector Search operations with Upstash services
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
import httpx
from upstash_redis import Redis
import base64
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpstashRedisClient:
    """Production-ready Upstash Redis client with REST API"""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        self.url = url or os.getenv('UPSTASH_REDIS_REST_URL')
        self.token = token or os.getenv('UPSTASH_REDIS_REST_TOKEN')

        if not self.url or not self.token:
            raise ValueError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN environment variables are required")

        self.client = Redis(url=self.url, token=self.token)
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify connection is working"""
        try:
            response = self.client.ping()
            if response == "PONG":
                logger.info("Upstash Redis connection verified successfully")
            else:
                raise ConnectionError("Failed to ping Upstash Redis")
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            raise

    # String Operations
    def set(self, key: str, value: Any, ex: Optional[int] = None, px: Optional[int] = None,
           nx: bool = False, xx: bool = False) -> bool:
        """Set a key-value pair with optional expiration"""
        try:
            kwargs = {}
            if ex is not None:
                kwargs['ex'] = ex
            if px is not None:
                kwargs['px'] = px
            if nx:
                kwargs['nx'] = nx
            if xx:
                kwargs['xx'] = xx

            result = self.client.set(key, value, **kwargs)
            return result == "OK" or result is True
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
            return self.client.expire(key, time) == 1
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
                return self.client.hset(name, mapping)
            else:
                return self.client.hset(name, {key: value})
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
            return self.client.hexists(name, key) == 1
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
            if count is None:
                return self.client.lpop(name)
            else:
                return self.client.lpop(name, count)
        except Exception as e:
            logger.error(f"Failed to lpop from list {name}: {e}")
            raise

    def rpop(self, name: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        """Pop value from right of list"""
        try:
            if count is None:
                return self.client.rpop(name)
            else:
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
            return set(self.client.smembers(name))
        except Exception as e:
            logger.error(f"Failed to get set members {name}: {e}")
            raise

    def sismember(self, name: str, value: Any) -> bool:
        """Check if value is in set"""
        try:
            return self.client.sismember(name, value) == 1
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
            # Convert mapping to the format expected by upstash-redis
            items = []
            for member, score in mapping.items():
                items.extend([score, member])

            return self.client.zadd(name, items, nx=nx, xx=xx, ch=ch, incr=incr)
        except Exception as e:
            logger.error(f"Failed to add to sorted set {name}: {e}")
            raise

    def zrange(self, name: str, start: int, end: int, desc: bool = False,
              withscores: bool = False) -> List:
        """Get range from sorted set"""
        try:
            return self.client.zrange(name, start, end, desc=desc, withscores=withscores)
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

    # Advanced Operations
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
            result = self.client.flushdb()
            return result == "OK"
        except Exception as e:
            logger.error(f"Failed to flush database: {e}")
            raise

    def ping(self) -> str:
        """Ping server"""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Failed to ping server: {e}")
            raise

class UpstashSearchClient:
    """Production-ready Upstash Vector Search client"""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        self.url = url or os.getenv('UPSTASH_SEARCH_REST_URL')
        self.token = token or os.getenv('UPSTASH_SEARCH_REST_TOKEN')

        if not self.url or not self.token:
            raise ValueError("UPSTASH_SEARCH_REST_URL and UPSTASH_SEARCH_REST_TOKEN environment variables are required")

        # Remove trailing slash if present
        self.url = self.url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify connection is working"""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.url}/info", headers=self.headers)
                if response.status_code == 200:
                    logger.info("Upstash Search connection verified successfully")
                else:
                    raise ConnectionError(f"Failed to connect to Upstash Search: {response.status_code}")
        except Exception as e:
            logger.error(f"Search connection verification failed: {e}")
            raise

    def upsert(self, vector_id: str, vector: List[float], metadata: Optional[Dict] = None) -> Dict:
        """Upsert a vector with optional metadata"""
        try:
            payload = {
                'id': vector_id,
                'vector': vector
            }
            if metadata:
                payload['metadata'] = metadata

            with httpx.Client() as client:
                response = client.post(
                    f"{self.url}/upsert",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to upsert vector {vector_id}: {e}")
            raise

    def upsert_batch(self, vectors: List[Dict]) -> Dict:
        """Upsert multiple vectors in batch"""
        try:
            payload = {'vectors': vectors}

            with httpx.Client() as client:
                response = client.post(
                    f"{self.url}/upsert-batch",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to upsert vector batch: {e}")
            raise

    def query(self, vector: List[float], top_k: int = 10,
             include_vectors: bool = False, include_metadata: bool = True,
             filter_expression: Optional[str] = None) -> Dict:
        """Query similar vectors"""
        try:
            payload = {
                'vector': vector,
                'topK': top_k,
                'includeVectors': include_vectors,
                'includeMetadata': include_metadata
            }

            if filter_expression:
                payload['filter'] = filter_expression

            with httpx.Client() as client:
                response = client.post(
                    f"{self.url}/query",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            raise

    def query_by_id(self, vector_id: str, top_k: int = 10,
                   include_vectors: bool = False, include_metadata: bool = True,
                   filter_expression: Optional[str] = None) -> Dict:
        """Query similar vectors by ID"""
        try:
            payload = {
                'id': vector_id,
                'topK': top_k,
                'includeVectors': include_vectors,
                'includeMetadata': include_metadata
            }

            if filter_expression:
                payload['filter'] = filter_expression

            with httpx.Client() as client:
                response = client.post(
                    f"{self.url}/query",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to query by vector ID {vector_id}: {e}")
            raise

    def fetch(self, vector_ids: List[str], include_vectors: bool = False,
             include_metadata: bool = True) -> Dict:
        """Fetch vectors by IDs"""
        try:
            payload = {
                'ids': vector_ids,
                'includeVectors': include_vectors,
                'includeMetadata': include_metadata
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.url}/fetch",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to fetch vectors {vector_ids}: {e}")
            raise

    def delete(self, vector_ids: List[str]) -> Dict:
        """Delete vectors by IDs"""
        try:
            payload = {'ids': vector_ids}

            with httpx.Client() as client:
                response = client.delete(
                    f"{self.url}/delete",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to delete vectors {vector_ids}: {e}")
            raise

    def delete_all(self) -> Dict:
        """Delete all vectors"""
        try:
            with httpx.Client() as client:
                response = client.delete(
                    f"{self.url}/delete-all",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to delete all vectors: {e}")
            raise

    def update(self, vector_id: str, vector: Optional[List[float]] = None,
              metadata: Optional[Dict] = None) -> Dict:
        """Update vector and/or metadata"""
        try:
            payload = {'id': vector_id}

            if vector is not None:
                payload['vector'] = vector
            if metadata is not None:
                payload['metadata'] = metadata

            with httpx.Client() as client:
                response = client.patch(
                    f"{self.url}/update",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to update vector {vector_id}: {e}")
            raise

    def range_query(self, vector: List[float], similarity_threshold: float,
                   max_results: int = 1000, include_vectors: bool = False,
                   include_metadata: bool = True) -> Dict:
        """Query vectors within similarity threshold"""
        try:
            payload = {
                'vector': vector,
                'threshold': similarity_threshold,
                'maxResults': max_results,
                'includeVectors': include_vectors,
                'includeMetadata': include_metadata
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.url}/range",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to perform range query: {e}")
            raise

    def info(self) -> Dict:
        """Get index information"""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.url}/info", headers=self.headers)
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            raise

    def reset(self) -> Dict:
        """Reset the index"""
        try:
            with httpx.Client() as client:
                response = client.post(f"{self.url}/reset", headers=self.headers)
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to reset index: {e}")
            raise
# =======================================================================


# =======================================================================
