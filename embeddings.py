"""Embedding backends.

Graded setting (default): SentenceTransformers `all-MiniLM-L6-v2`, free + local.
`EMBED_BACKEND=hash` swaps in a tiny deterministic hashing embedder that needs
no model download. It exists ONLY so the pipeline can be smoke-tested on a
machine that cannot reach huggingface.co - transcripts produced with it are not
the graded ones, and every results file records which embedder was used.
"""
import hashlib
import re

import numpy as np

import config

_STOP = set(
    "a an the and or of to in on at for from by with is are was were be been it its this that these those "
    "do does did can could should would will may might must as if so than then there their they them we you "
    "your our i me my what which who whom how when where why not no any all each per into out up".split()
)


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokens(text: str) -> list[str]:
    """Shared lightweight tokenizer (also used by guardrails / judge / mock LLM)."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


class HashingEmbedder:
    dim = 384
    name = "hash-384 (offline smoke-test embedder, NOT SentenceTransformers)"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokens(text)
        feats = [(t, 1.0) for t in toks] + [(a + "_" + b, 0.6) for a, b in zip(toks, toks[1:])]
        for f, w in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += w * (1.0 if (h >> 64) & 1 else -1.0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if len(texts) else np.zeros((0, self.dim), np.float32)


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))


_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if config.EMBED_BACKEND == "hash":
            _EMBEDDER = HashingEmbedder()
        else:
            try:
                _EMBEDDER = STEmbedder(config.ST_MODEL_NAME)
            except Exception as e:  # never silently downgrade the graded setting
                raise RuntimeError(
                    "Could not load the SentenceTransformers model "
                    f"'{config.ST_MODEL_NAME}'. Run `pip install sentence-transformers` and make sure you "
                    "are online for the first run (the ~90 MB model is cached afterwards). "
                    "For an offline smoke test only, set EMBED_BACKEND=hash.\n"
                    f"Original error: {e}"
                ) from e
    return _EMBEDDER
