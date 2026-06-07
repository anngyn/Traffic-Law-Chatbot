# memory/cache.py — response cache (Redis with in-memory fallback).
import hashlib
import os


def make_key(*parts: str) -> str:
    return "chat:" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._mem: dict[str, str] = {}
        self._redis = None
        url = os.getenv("REDIS_URL")
        if url:
            try:
                import redis

                self._redis = redis.Redis.from_url(url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get(self, key: str):
        if self._redis:
            try:
                return self._redis.get(key)
            except Exception:
                return None
        return self._mem.get(key)

    def set(self, key: str, value: str):
        if self._redis:
            try:
                self._redis.set(key, value, ex=self.ttl)
                return
            except Exception:
                pass
        self._mem[key] = value


cache = Cache()
