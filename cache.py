"""Response cache + call counters (Task 16)."""
import re
import time
from collections import Counter

COUNTERS: Counter = Counter()  # retrieval_calls, generation_calls, tool_calls, cache_hits, cache_misses


def normalize_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s₹]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


class ResponseCache:
    """In-memory cache keyed by normalized query text (+ strategy)."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def key(self, query: str, strategy: str) -> str:
        return f"{strategy}::{normalize_query(query)}"

    def get(self, query: str, strategy: str):
        k = self.key(query, strategy)
        if k in self._store:
            COUNTERS["cache_hits"] += 1
            return dict(self._store[k], cache_hit=True)
        COUNTERS["cache_misses"] += 1
        return None

    def set(self, query: str, strategy: str, value: dict) -> None:
        self._store[self.key(query, strategy)] = dict(value)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self):
        return len(self._store)


GROUNDED_CACHE = ResponseCache()


def snapshot() -> dict:
    return dict(COUNTERS)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000
