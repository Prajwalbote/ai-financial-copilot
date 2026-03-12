"""
utils/cache.py — Response Caching
===================================
WHY CACHING?
    Every RAG query loads the LLM and runs inference.
    On CPU this takes 5-10 seconds per query.
    
    If 10 users ask "What is Apple's revenue?" we don't
    need to run the LLM 10 times — cache the first result
    and return it instantly for subsequent requests.

    This is how production AI systems handle scale.

WHAT YOU LEARN:
    - In-memory caching pattern
    - Cache invalidation strategy
    - TTL (Time To Live) for cache entries
    - How to reduce LLM inference costs
"""

import time
import hashlib
import json
from typing import Optional
from utils.logger import get_logger

log = get_logger(__name__)


class ResponseCache:
    """
    Simple in-memory cache for RAG responses.

    WHY IN-MEMORY vs REDIS?
    Redis is better for production (persistent, shared
    across multiple servers). In-memory is simpler and
    works perfectly for single-server deployments.

    For scale: swap this class internals for Redis,
    keep the same interface. Nothing else changes.

    Usage:
        cache = ResponseCache(ttl_seconds=3600)
        key = cache.make_key("What is revenue?", "apple_10k")

        cached = cache.get(key)
        if cached:
            return cached

        result = expensive_rag_call()
        cache.set(key, result)
        return result
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Args:
            ttl_seconds: How long to keep cached responses
                         3600 = 1 hour (good for financial docs
                         that don't change often)
        """
        self._cache = {}          # key → {value, expires_at}
        self.ttl_seconds = ttl_seconds
        self.hits = 0             # Cache hits counter
        self.misses = 0           # Cache misses counter
        log.info(
            f"ResponseCache initialized | "
            f"ttl={ttl_seconds}s"
        )

    def make_key(self, query: str, index_name: str) -> str:
        """
        Create a unique cache key from query + index.

        WHY HASH?
        Queries can be long strings. Using the full string
        as a key wastes memory. MD5 hash is fast and gives
        a fixed-length 32-char key.

        Note: MD5 is NOT for security here, just for
        creating consistent short keys.
        """
        raw_key = f"{query.lower().strip()}:{index_name}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        """
        Get a cached response if it exists and hasn't expired.

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self.misses += 1
            return None

        entry = self._cache[key]

        # Check if expired
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            self.misses += 1
            log.debug(f"Cache expired for key: {key[:8]}...")
            return None

        self.hits += 1
        log.info(
            f"Cache HIT | "
            f"key={key[:8]}... | "
            f"hits={self.hits} | "
            f"misses={self.misses}"
        )
        return entry["value"]

    def set(self, key: str, value: dict):
        """
        Store a response in the cache.

        Args:
            key:   Cache key from make_key()
            value: Response dict to cache
        """
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + self.ttl_seconds,
            "cached_at": time.time()
        }
        log.debug(
            f"Cache SET | "
            f"key={key[:8]}... | "
            f"total_entries={len(self._cache)}"
        )

    def invalidate(self, key: str):
        """Remove a specific entry from cache."""
        if key in self._cache:
            del self._cache[key]
            log.info(f"Cache invalidated: {key[:8]}...")

    def clear(self):
        """Clear all cached entries."""
        count = len(self._cache)
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        log.info(f"Cache cleared | removed {count} entries")

    def get_stats(self) -> dict:
        """
        Get cache performance statistics.

        WHY TRACK STATS?
        Cache hit rate tells you if caching is working.
        Low hit rate = queries are too unique (caching less useful)
        High hit rate = caching saving lots of compute
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "total_entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 1),
            "ttl_seconds": self.ttl_seconds
        }


# Singleton cache instance
_cache_instance = None


def get_cache() -> ResponseCache:
    """Get the singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache(ttl_seconds=3600)
    return _cache_instance