"""RAG core: two chunking strategies, two ChromaDB collections, grounded generation,
threshold calibration and precision/recall evaluation (Part 1, Tasks 3-5)."""
import json
import re
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

import config
from cache import COUNTERS, GROUNDED_CACHE
from embeddings import get_embedder
from kb_docs import KB_DOCS
from mock_llm import FALLBACK, generate_grounded, split_sentences

STRATEGIES = ("fixed", "sentence")


# ----------------------------------------------------------------- chunkers
def chunk_fixed(text: str, size: int = config.CHUNK_SIZE_CHARS, overlap: int = config.CHUNK_OVERLAP_CHARS):
    """Fixed-size windows with overlap, snapped to word boundaries. -> [(start, end, text)]"""
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            sp = text.rfind(" ", start + size // 2, end)
            end = sp if sp != -1 else end
        piece = text[start:end].strip()
        if piece:
            out.append((start, end, piece))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        sp = text.find(" ", nxt)  # start the next window on a word boundary
        start = (sp + 1) if sp != -1 and sp < end else nxt
    return out


def chunk_sentences(text: str):
    """One chunk per sentence. -> [(start, end, text)]"""
    out, cursor = [], 0
    for s in split_sentences(text):
        i = text.find(s, cursor)
        out.append((i, i + len(s), s))
        cursor = i + len(s)
    return out


CHUNKERS = {"fixed": chunk_fixed, "sentence": chunk_sentences}


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # cosine similarity
    strategy: str
    clean_start: bool = True  # chunk begins at a sentence boundary
    clean_end: bool = True    # chunk ends with sentence-final punctuation


# ----------------------------------------------------------------- store
class RagStore:
    def __init__(self, docs=None, embedder=None):
        self.docs = {d["doc_id"]: dict(d) for d in (docs or KB_DOCS)}
        self.embedder = embedder or get_embedder()
        self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False, allow_reset=True))
        suffix = uuid.uuid4().hex[:8]
        self.collections = {
            s: self.client.get_or_create_collection(f"kb_{s}_{suffix}", metadata={"hnsw:space": "cosine"})
            for s in STRATEGIES
        }
        self.thresholds: dict[str, float] = {}
        self.recommended: str | None = None
        for d in self.docs.values():
            self._index_doc(d)

    def _index_doc(self, doc: dict) -> int:
        n = 0
        for strat, chunker in CHUNKERS.items():
            chunks = chunker(doc["text"])
            if not chunks:
                continue
            texts = [c[2] for c in chunks]
            ids = [f"{doc['doc_id']}::{strat}::{i}" for i in range(len(chunks))]
            metas = [
                {
                    "doc_id": doc["doc_id"], "strategy": strat, "start": c[0], "end": c[1],
                    "clean_start": (not doc["text"][: c[0]].rstrip()) or doc["text"][: c[0]].rstrip()[-1] in ".!?",
                    "clean_end": c[2].rstrip().endswith((".", "!", "?")),
                }
                for c in chunks
            ]
            self.collections[strat].upsert(
                ids=ids, documents=texts, metadatas=metas, embeddings=self.embedder.encode(texts).tolist()
            )
            n += len(chunks)
        return n

    def add_document(self, doc_id: str, text: str, topic: str = "user-added") -> dict:
        doc = {"doc_id": doc_id, "topic": topic, "text": text}
        self.docs[doc_id] = doc
        n = self._index_doc(doc)
        GROUNDED_CACHE.clear()  # KB changed -> cached answers may be stale
        return {"doc_id": doc_id, "chunks_indexed": n}

    def count(self, strategy: str) -> int:
        return self.collections[strategy].count()

    def retrieve(self, query: str, strategy: str, k: int = config.TOP_K) -> list[Hit]:
        COUNTERS["retrieval_calls"] += 1
        res = self.collections[strategy].query(
            query_embeddings=self.embedder.encode([query]).tolist(), n_results=min(k, self.count(strategy))
        )
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(
                Hit(cid, meta["doc_id"], doc, round(1.0 - float(dist), 4), strategy,
                    bool(meta.get("clean_start", True)), bool(meta.get("clean_end", True)))
            )
        return hits

    # ---------------------------------------------------- calibration / eval
    def calibrate(self, in_scope: list[str] = None, out_of_scope: list[str] = None) -> dict:
        in_scope = in_scope or [q for q, _ in EVAL_QUERIES] + EXTRA_IN_SCOPE
        out_of_scope = out_of_scope or OUT_OF_SCOPE_QUERIES
        result = {}
        for strat in STRATEGIES:
            ins = [(q, self.retrieve(q, strat, 1)[0].score) for q in in_scope]
            outs = [(q, self.retrieve(q, strat, 1)[0].score) for q in out_of_scope]
            lo_in = min(s for _, s in ins)
            hi_out = max(s for _, s in outs)
            separable = hi_out < lo_in
            if separable:
                thr = round((lo_in + hi_out) / 2, 3)
            else:  # clusters overlap: pick the cut that misclassifies the fewest queries
                cuts = sorted({s for _, s in ins + outs})
                thr = max(cuts, key=lambda c: sum(s >= c for _, s in ins) + sum(s < c for _, s in outs))
                thr = round(thr - 1e-3, 3)
            result[strat] = {
                "in_scope": [(q, s) for q, s in ins],
                "out_of_scope": [(q, s) for q, s in outs],
                "min_in_scope": lo_in,
                "max_out_of_scope": hi_out,
                "clusters_separate": separable,
                "threshold": thr,
            }
            self.thresholds[strat] = thr
        return result

    def evaluate_chunking(self, k: int = config.TOP_K, queries=None) -> dict:
        queries = queries or EVAL_QUERIES[:8]
        out = {}
        for strat in STRATEGIES:
            rows = []
            for q, relevant in queries:
                hits = self.retrieve(q, strat, k)
                retrieved_docs = list(dict.fromkeys(h.doc_id for h in hits))  # map to parent docs + dedup
                rel = set(relevant)
                tp = len(rel & set(retrieved_docs))
                p = tp / len(retrieved_docs) if retrieved_docs else 0.0
                r = tp / len(rel)
                rows.append(
                    {
                        "query": q,
                        "relevant": sorted(rel),
                        "retrieved_chunks": [h.chunk_id for h in hits],
                        "retrieved_docs": retrieved_docs,
                        "tp": tp,
                        "precision": p,
                        "recall": r,
                        "precision_arith": f"{tp}/{len(retrieved_docs)} = {p:.2f}",
                        "recall_arith": f"{tp}/{len(rel)} = {r:.2f}",
                    }
                )
            mp = sum(r["precision"] for r in rows) / len(rows)
            mr = sum(r["recall"] for r in rows) / len(rows)
            f1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
            out[strat] = {"rows": rows, "macro_precision": mp, "macro_recall": mr, "macro_f1": f1}
        f_fixed, f_sent = out["fixed"]["macro_f1"], out["sentence"]["macro_f1"]
        self.recommended = "fixed" if f_fixed > f_sent else "sentence"  # tie -> finer-grained sentence chunks
        other = "sentence" if self.recommended == "fixed" else "fixed"
        a, b = out[self.recommended], out[other]
        out["recommended"] = self.recommended
        out["recommendation_text"] = (
            f"I would deploy the {self.recommended}-chunk collection: it scored macro precision "
            f"{a['macro_precision']:.2f} / recall {a['macro_recall']:.2f} (F1 {a['macro_f1']:.2f}) versus "
            f"{b['macro_precision']:.2f} / {b['macro_recall']:.2f} (F1 {b['macro_f1']:.2f}) for the {other}-chunk "
            f"collection on the same {len(queries)} queries at k={k}. "
            + (
                "Whole-sentence chunks keep each fact self-contained, which keeps retrieved context clean."
                if self.recommended == "sentence"
                else "Overlapping windows keep neighbouring sentences together, so related facts are retrieved as one unit."
            )
        )
        return out

    def ensure_ready(self):
        """Calibrate thresholds and pick the deployed strategy once (idempotent)."""
        if not self.thresholds:
            self.calibrate()
        if not self.recommended:
            self.evaluate_chunking()


# ----------------------------------------------------------------- query sets
# (query, relevant parent docs)
EVAL_QUERIES = [
    ("How fast must we reply to a P1 ticket?", ["kb-02"]),
    ("When should a Tier 1 agent escalate a ticket to Tier 2?", ["kb-03"]),
    ("How long does a refund to the original payment method take?", ["kb-04"]),
    ("Can an agent offer a phone callback to a customer?", ["kb-05"]),
    ("Is night support available on public holidays?", ["kb-06"]),
    ("What happens if a customer raises the same complaint again within two weeks?", ["kb-07"]),
    ("How much service credit does a customer get when the SLA is breached?", ["kb-08", "kb-02"]),
    ("How long does the satisfaction survey stay open after a ticket is resolved?", ["kb-09"]),
    ("How are VIP customer tickets prioritised?", ["kb-10"]),
    ("How often are outage updates posted?", ["kb-11"]),
    ("How long are resolved tickets kept before deletion?", ["kb-12"]),
    ("How is a ticket classified as P1 priority?", ["kb-01"]),
]
EXTRA_IN_SCOPE = ["What is the compensation cap for a poor experience?"]
OUT_OF_SCOPE_QUERIES = [
    "What is the capital of Australia?",
    "Give me a recipe for paneer butter masala.",
    "Who won the cricket world cup in 2011?",
    "How do I train a neural network in PyTorch?",
]


# ----------------------------------------------------------------- grounded generation
def usable_context(hit: Hit) -> str:
    """Drop sentence fragments created by fixed-size windows so the answer quotes whole sentences only."""
    sents = split_sentences(hit.text)
    if not hit.clean_start and len(sents) > 1:
        sents = sents[1:]
    if not hit.clean_end and len(sents) > 1:
        sents = sents[:-1]
    return " ".join(sents)


_STORE: RagStore | None = None


def get_store() -> RagStore:
    global _STORE
    if _STORE is None:
        _STORE = RagStore()
        _STORE.ensure_ready()
    return _STORE


def grounded_answer(query: str, strategy: str | None = None, k: int = config.TOP_K, store: RagStore | None = None) -> dict:
    """Retrieve top-k chunks and answer ONLY from them; fall back below the calibrated threshold."""
    store = store or get_store()
    store.ensure_ready()
    strategy = strategy or store.recommended
    cached = GROUNDED_CACHE.get(query, strategy)
    if cached is not None:
        return cached
    hits = store.retrieve(query, strategy, k)
    top = hits[0].score if hits else 0.0
    thr = store.thresholds[strategy]
    if top < thr:
        result = {"answer": FALLBACK, "grounded": False, "top_score": top, "threshold": thr, "sources": [], "contexts": []}
    else:
        contexts = [h.text for h in hits]
        result = {
            "answer": generate_grounded(query, [usable_context(h) for h in hits]),
            "grounded": True,
            "top_score": top,
            "threshold": thr,
            "sources": list(dict.fromkeys(h.doc_id for h in hits)),
            "contexts": contexts,
        }
    result.update(strategy=strategy, cache_hit=False)
    GROUNDED_CACHE.set(query, strategy, result)
    return result
