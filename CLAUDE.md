# Project context (read first)

IITR capstone, **Support domain / Ola track**: a CrewAI domain support agent (RAG + 2 tools + 3-agent crew + Autogen review + FastAPI).
Deliverable = ONE public GitHub repo (this folder). The owner is also preparing for GATE, so be efficient and don't over-engineer.

## State
All 16 tasks were implemented and passed end to end, plus 18 pytest tests, but ONLY with the offline smoke-test embedder
(`EMBED_BACKEND=hash`), because the sandbox that built this could not reach huggingface.co. The real SentenceTransformers run has NOT happened yet.

## Your job now (Windows, PowerShell)
1. `.\setup.ps1` (Python 3.10-3.13; 3.14 is unsupported). The protobuf conflict warning from autogen is expected; setup restores protobuf>=6.33.5.
2. `$env:PYTHONUTF8="1"; python run_all.py` - needs internet once for the ~90 MB model. Fix any failure.
3. `python make_readme.py` - writes README.md from transcripts/results.json (it refuses hash-embedder results).
4. `pytest -q`. Then commit and push to a public GitHub repo.

## Rules that matter (these are graded)
- Graded mode is `MOCK_LLM=true`: zero API keys, zero network at run time. Never set EMBED_BACKEND=hash for the graded run.
- `config.py` must stay the FIRST import in every entry point (it sets CREWAI_DISABLE_TELEMETRY / OTEL_SDK_DISABLED before crewai loads).
- Never hand-edit dataset records; to change the escalated% band, change SEED/weights in `dataset.py` (currently seed 13, 60 records).
- The "I don't know" threshold is measured by `RagStore.calibrate()`, never a preset 0.5/0.6/0.7. If the real-embedder clusters overlap, the code picks the best cut and the README reports `clusters_separate=False`; tune KB wording or queries instead of faking numbers.
- Only the Lookup Agent may hold `check_support_ticket_status` (enforced in `crew.make_agent`).
- Logs must never contain raw phone numbers (masking happens before logging; run_all asserts it).
- No images/PDFs/slides anywhere in the repo.
- Brief pitfalls handled in `mock_llm.MockCrewLLM`: never search the whole conversation for "Observation:", and dispatch tools by argument schema, not tool name.

## Map
dataset.py, kb_docs.py, embeddings.py, rag.py (Part 1) | tools.py, mock_llm.py, crew.py, memory.py, guardrails.py, schemas.py (Part 2) |
app.py, logging_utils.py, pipeline.py, evaluation.py (Part 3) | review.py, governance.py, cache.py (Part 4) | run_all.py runs every demo.
