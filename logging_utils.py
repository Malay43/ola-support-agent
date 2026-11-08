"""ELK-style structured logging: ONE JSON-Lines entry per request, with trace id + timings.
Everything is passed through the same PII masker the model sees before it reaches disk (Task 12)."""
import json
import threading
import time
import uuid

import config
from guardrails import mask_obj

_LOCK = threading.Lock()


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def log_request(entry: dict, path=None) -> dict:
    entry = dict(entry)
    entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    entry = mask_obj(entry)  # defence in depth: nothing PII-shaped ever reaches the log
    with _LOCK, open(path or config.REQUEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
