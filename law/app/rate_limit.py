"""Redis-backed fixed-window rate limiting."""

from __future__ import annotations

import time


class DistributedRateLimiter:
    def __init__(self, redis_client, limit: int, window_seconds: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window_seconds = window_seconds

    def allow(self, identity: str) -> bool:
        bucket = int(time.time()) // self.window_seconds
        key = f"rag:rate:{identity}:{bucket}"
        pipeline = self.redis.pipeline(transaction=True)
        pipeline.incr(key)
        pipeline.expire(key, self.window_seconds + 5)
        count, _ = pipeline.execute()
        return int(count) <= self.limit
