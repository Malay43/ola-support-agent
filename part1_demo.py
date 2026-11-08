"""Part 1 demo: dataset report, chunk counts, calibration, grounded answers, P/R comparison."""
import json

import config
import dataset
from embeddings import get_embedder
from rag import EVAL_QUERIES, RagStore, grounded_answer


def run() -> dict:
    emb = get_embedder()
    print(f"[embedder] {emb.name}\n")
    print("=== TASK 1: dataset ===")
    print(dataset.report())

    store = RagStore(embedder=emb)
    print("\n=== TASK 3: chunking + indexing (two separate ChromaDB collections) ===")
    for s in ("fixed", "sentence"):
        print(f"  collection[{s}]: {store.count(s)} chunks from {len(store.docs)} documents")
    probe = "What is the SLA for a P2 ticket?"
    for s in ("fixed", "sentence"):
        top = store.retrieve(probe, s, 1)[0]
        print(f"  sample query on [{s}] -> {top.chunk_id} (cos={top.score:.3f}): {top.text[:90]}...")

    print("\n=== TASK 4: threshold calibration (measured, not a preset) ===")
    cal = store.calibrate()
    for s, c in cal.items():
        print(f"  [{s}] in-scope top-1 cosines:")
        for q, sc in c["in_scope"]:
            print(f"      {sc:.3f}  {q}")
        print(f"  [{s}] out-of-scope top-1 cosines:")
        for q, sc in c["out_of_scope"]:
            print(f"      {sc:.3f}  {q}")
        print(f"  [{s}] min(in-scope)={c['min_in_scope']:.3f}  max(out-of-scope)={c['max_out_of_scope']:.3f}  "
              f"separable={c['clusters_separate']}  -> threshold={c['threshold']}")

    print("\n=== TASK 5: precision/recall per collection (document level, k=3) ===")
    ev = store.evaluate_chunking()
    for s in ("fixed", "sentence"):
        print(f"\n  -- collection[{s}] --")
        for r in ev[s]["rows"]:
            print(f"  Q: {r['query']}")
            print(f"     relevant={r['relevant']} retrieved_docs={r['retrieved_docs']}")
            print(f"     precision = {r['precision_arith']}   recall = {r['recall_arith']}")
        print(f"  macro precision={ev[s]['macro_precision']:.3f} recall={ev[s]['macro_recall']:.3f} F1={ev[s]['macro_f1']:.3f}")
    print("\n  Recommendation:", ev["recommendation_text"])

    print(f"\n=== TASK 4: grounded generation on the deployed [{store.recommended}] collection ===")
    answers = []
    for q, _ in EVAL_QUERIES[:6]:
        a = grounded_answer(q, store=store)
        answers.append({"query": q, **{k: a[k] for k in ("answer", "grounded", "top_score", "sources")}})
        print(f"  Q: {q}\n     top1={a['top_score']:.3f} grounded={a['grounded']} sources={a['sources']}\n     A: {a['answer']}")
    oos = grounded_answer("What is the capital of Australia?", store=store)
    answers.append({"query": "What is the capital of Australia?", **{k: oos[k] for k in ("answer", "grounded", "top_score", "sources")}})
    print(f"  Q (out of scope): What is the capital of Australia?\n     top1={oos['top_score']:.3f} grounded={oos['grounded']}\n     A: {oos['answer']}")
    assert oos["grounded"] is False

    results = {
        "embedder": emb.name, "dataset": dataset.summarize(dataset.SUPPORT_TICKETS), "seed": dataset.SEED,
        "chunk_counts": {s: store.count(s) for s in ("fixed", "sentence")},
        "calibration": cal, "chunk_eval": ev, "recommended": store.recommended, "answers": answers,
    }
    (config.TRANSCRIPT_DIR / "part1_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    run()
