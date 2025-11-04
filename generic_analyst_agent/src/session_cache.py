"""Session Caching: Cache query results to avoid redundant API calls.

This module provides a simple in-memory cache with TTL (time-to-live) for 
storing agent responses. Helps reduce API costs and improve response time
for repeated queries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SessionCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """Initialize the cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default 1 hour)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        
    def _generate_key(self, question: str, dataset_signature: str) -> str:
        """Generate a cache key from question and dataset signature.
        
        Args:
            question: The user question (normalized)
            dataset_signature: A signature of the dataset (hash of columns/shape)
            
        Returns:
            Cache key string
        """
        # Normalize question (lowercase, strip whitespace)
        normalized = question.lower().strip()
        
        # Combine with dataset signature
        combined = f"{dataset_signature}:{normalized}"
        
        # Hash to fixed-length key
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, question: str, dataset_signature: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available and not expired.
        
        Args:
            question: The user question
            dataset_signature: Signature of the current dataset
            
        Returns:
            Cached result dict or None if not found/expired
        """
        key = self._generate_key(question, dataset_signature)
        
        if key not in self.cache:
            logger.debug(f"Cache MISS: {question[:50]}")
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            logger.debug(f"Cache EXPIRED: {question[:50]}")
            del self.cache[key]
            return None
        
        logger.info(f"Cache HIT: {question[:50]}")
        return entry["result"]
    
    def set(self, question: str, dataset_signature: str, result: Dict[str, Any]) -> None:
        """Store result in cache.
        
        Args:
            question: The user question
            dataset_signature: Signature of the current dataset
            result: The agent result to cache
        """
        key = self._generate_key(question, dataset_signature)
        
        self.cache[key] = {
            "timestamp": time.time(),
            "result": result,
            "question": question,  # For debugging
        }
        
        logger.debug(f"Cache SET: {question[:50]}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry["timestamp"] > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache stats (size, oldest entry, etc.)
        """
        if not self.cache:
            return {
                "size": 0,
                "oldest_age_seconds": 0,
                "newest_age_seconds": 0,
            }
        
        current_time = time.time()
        ages = [current_time - entry["timestamp"] for entry in self.cache.values()]
        
        return {
            "size": len(self.cache),
            "oldest_age_seconds": max(ages),
            "newest_age_seconds": min(ages),
            "ttl_seconds": self.ttl_seconds,
        }


def generate_dataset_signature(df) -> str:
    """Generate a signature for a dataframe based on shape and columns.
    
    This is used to invalidate cache when dataset changes.
    
    Args:
        df: The dataframe
        
    Returns:
        Signature string
    """
    try:
        import pandas as pd
        
        # Use shape + column names + dtypes as signature
        components = [
            str(df.shape),
            ",".join(df.columns.tolist()),
            ",".join(df.dtypes.astype(str).tolist()),
        ]
        signature_str = "|".join(components)
        
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Failed to generate dataset signature: {e}")
        return "unknown"


# Global cache instance (can be customized per app)
_global_cache: Optional[SessionCache] = None


def get_cache(ttl_seconds: int = 3600) -> SessionCache:
    """Get or create the global cache instance.
    
    Args:
        ttl_seconds: TTL for cache entries (only used on first call)
        
    Returns:
        SessionCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = SessionCache(ttl_seconds=ttl_seconds)
    return _global_cache
