"""Generate README.md from the REAL numbers in transcripts/results.json.

    python run_all.py && python make_readme.py

Refuses to write a README from an offline smoke-test run (EMBED_BACKEND=hash) unless --allow-smoke is given,
so you can't accidentally submit numbers that did not come from SentenceTransformers.
"""
import json
import sys

import config
import dataset
import tools

R = json.loads((config.TRANSCRIPT_DIR / "results.json").read_text())
p1, emb = R["part1"], R["meta"]["embedder"]
smoke = "hash" in emb.lower()
if smoke and "--allow-smoke" not in sys.argv:
    sys.exit("These results came from the offline hash embedder, not SentenceTransformers. "
             "Run `python run_all.py` WITHOUT EMBED_BACKEND=hash first (needs internet once to download the model).")

banner = ("> **WARNING: smoke-test run with the offline hash embedder - NOT valid for submission.**\n\n" if smoke else "")
ds = p1["dataset"]
cal = p1["calibration"]
ev = p1["chunk_eval"]
rec = p1["recommended"]


def cal_table(strat):
    c = cal[strat]
    rows = ["| group | top-1 cosine | query |", "|---|---|---|"]
    rows += [f"| in-scope | {s:.3f} | {q} |" for q, s in c["in_scope"]]
    rows += [f"| out-of-scope | {s:.3f} | {q} |" for q, s in c["out_of_scope"]]
    return "\n".join(rows)


def pr_table(strat):
    rows = ["| query | relevant doc(s) | retrieved docs (deduped) | precision | recall |", "|---|---|---|---|---|"]
    for r in ev[strat]["rows"]:
        rows.append(f"| {r['query']} | {', '.join(r['relevant'])} | {', '.join(r['retrieved_docs'])} | {r['precision_arith']} | {r['recall_arith']} |")
    e = ev[strat]
    rows.append(f"| **macro average** | | | **{e['macro_precision']:.3f}** | **{e['macro_recall']:.3f}** (F1 {e['macro_f1']:.3f}) |")
    return "\n".join(rows)


ev_rows = ["| # | kind | accuracy | grounding | completeness | safety | question |", "|---|---|---|---|---|---|---|"]
for r in R["task13"]["rows"]:
    ev_rows.append(f"| {r['n']} | {r['kind']} | {r['accuracy']} | {r['grounding']} | {r['completeness']} | {r['safety']} | {r['question'][:80]} |")
a = R["task13"]["averages"]
ev_rows.append(f"| | **AVERAGE (15)** | **{a['accuracy']}** | **{a['grounding']}** | **{a['completeness']}** | **{a['safety']}** | |")

t7 = "\n".join(f"- `{r['q']}` -> `rag_lookup` x{r['tool_calls_rag']}, `check_support_ticket_status` x{r['tool_calls_lookup']}" for r in R["task7"])
t14 = R["task14"]
t16 = R["task16"]

README = f"""# Ola Domain Support Agent (CrewAI + Autogen review + FastAPI)

**Track completed: Ola (Business Operations / Customer Support)**  - domain `Support`.

{banner}Everything in this repository runs under `MOCK_LLM` (the default) with **zero API keys and zero network access** after the
one-time download of the free SentenceTransformers model (`{config.ST_MODEL_NAME}`). Embedder used for the numbers below: **{emb}**.

**Telemetry:** I set `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true` (plus `CREWAI_TRACING_ENABLED=false`,
`ANONYMIZED_TELEMETRY=False` for ChromaDB) in `config.py`, which every entry point imports *before* `crewai`/`chromadb`, so
`crew.kickoff()` makes no outbound call. (`run_all.py` prints their values at the start of the run.)

The knowledge-base policies are **fictional and illustrative**, written for this project; they are not Ola's real internal policies.

## How to run

```bash
bash setup.sh                  # venv + pinned dependencies (Windows: .\\setup.ps1; see protobuf note inside)
source venv/bin/activate   # Windows: venv\\Scripts\\Activate.ps1
python run_all.py              # runs every task's demo, writes transcripts/*.txt (about a minute)
python make_readme.py          # regenerates this README from the real numbers
uvicorn app:app --port 8000    # optional: serve the API (POST /ask, POST /add-document, WS /ws/chat)
pytest -q                      # optional: unit tests
```

## Repository map

| file | what it is |
|---|---|
| `dataset.py` | seeded ticket generator + validation (Task 1) |
| `kb_docs.py` | 12 knowledge-base documents (Task 2) |
| `embeddings.py`, `rag.py` | SentenceTransformers, two chunkers, two ChromaDB collections, grounded generation, calibration, precision/recall (Tasks 3-5) |
| `tools.py` | `check_support_ticket_status` + escalation score; CrewAI tool wrappers (Task 6) |
| `mock_llm.py` | `MockCrewLLM(BaseLLM)` for CrewAI + deterministic generator (Task 7 / MOCK_LLM) |
| `crew.py` | Retrieval Agent, Lookup Agent, Response Composer; `.kickoff()` (Tasks 7, 9) |
| `memory.py` | `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` (Task 8) |
| `guardrails.py` | PII masking, prompt-injection detection, groundedness guard (Task 10) |
| `app.py`, `logging_utils.py`, `pipeline.py` | FastAPI (2 HTTP + 1 WebSocket), JSONL logging, request pipeline (Tasks 11-12) |
| `evaluation.py` | LLM-as-judge on 15 queries (Task 13) |
| `review.py` | Autogen `RoundRobinGroupChat` review stage (Task 14) |
| `governance.py`, `cache.py` | least autonomy, risk class, token budget; response cache (Tasks 15-16) |
| `run_all.py`, `transcripts/` | runs everything and stores the transcript for every task |

## Part 1 - Dataset design and RAG core

### Task 1 - dataset design choices (reproducible)

- `random.Random(SEED)` is the only randomness source: **SEED = {p1['seed']}**, **N = {ds['n']}** records, generated by `dataset.py`.
- Category weights `{dict(zip(dataset.CATEGORIES, dataset.CATEGORY_WEIGHTS))}`
- Status weights `{dict(zip(dataset.STATUSES, dataset.STATUS_WEIGHTS))}`
- `escalated = True` for every ticket in status `Escalated`, otherwise with probability {dataset.BASE_ESCALATION_PROB}.
- **Amount range** (`resolution_time_hours`): lognormal around a per-category median {dataset.CATEGORY_MEDIAN_HOURS}, clamped to
  {dataset.RESOLUTION_HOURS_RANGE} hours - quick inquiries and account-access issues close in minutes to hours, billing in about a working day, and
  product defects take days; 96 h is the P4 SLA ceiling. For unresolved tickets it is the hours elapsed so far.
- `days_since_created` is uniform on 0-30.
- No record was hand-edited; seed 13 was chosen from a seed sweep for a balanced draw.

Counts per category: `{ds['per_category']}`  
Counts per status: `{ds['per_status']}`  
`escalated=True`: **{ds['escalated_pct']}%** (required band 10-30%).

### Tasks 2-3 - knowledge base and the two ChromaDB collections

12 documents in `kb_docs.py` (4-5 sentences each) cover all 12 required topics. Chunking: fixed-size
({config.CHUNK_SIZE_CHARS} chars, {config.CHUNK_OVERLAP_CHARS} overlap, word-boundary snapped) and sentence-based; each strategy is embedded and
`upsert`ed into its own collection: **fixed = {p1['chunk_counts']['fixed']} chunks, sentence = {p1['chunk_counts']['sentence']} chunks**.

### Task 4 - grounded generation and the measured "I don't know" threshold

I measured top-1 cosine similarity for {len(cal['fixed']['in_scope'])} in-scope and {len(cal['fixed']['out_of_scope'])} out-of-scope queries (not a preset 0.5/0.6/0.7).

**Collection `fixed`:** min in-scope = {cal['fixed']['min_in_scope']:.3f}, max out-of-scope = {cal['fixed']['max_out_of_scope']:.3f}, clusters separate = {cal['fixed']['clusters_separate']} -> **threshold = {cal['fixed']['threshold']}**

{cal_table('fixed')}

**Collection `sentence`:** min in-scope = {cal['sentence']['min_in_scope']:.3f}, max out-of-scope = {cal['sentence']['max_out_of_scope']:.3f}, clusters separate = {cal['sentence']['clusters_separate']} -> **threshold = {cal['sentence']['threshold']}**

{cal_table('sentence')}

The deployed collection is **`{rec}`** (see Task 5); its threshold is **{cal[rec]['threshold']}**. Six in-scope queries plus one out-of-scope query that triggers
the fallback are in `transcripts/part1_dataset_rag.txt`.

### Task 5 - precision / recall per collection (document level, top-k = 3, chunks mapped to parent docs and de-duplicated)

**Collection `fixed`**

{pr_table('fixed')}

**Collection `sentence`**

{pr_table('sentence')}

**Recommendation:** {ev['recommendation_text']}

## Part 2 - CrewAI orchestration

### Task 6 - `check_support_ticket_status` and the designed escalation score

{R['task6']['justification']}

### Task 7 - the crew (`.kickoff()`)

Retrieval Agent (holds `rag_lookup`), Lookup Agent (holds `check_support_ticket_status`), Response Composer (no tools). Tool invocations per query:

{t7}

### Tasks 8-10 - memory, structured output, guardrails

- **Memory:** `RunnableWithMessageHistory` over `InMemoryChatMessageHistory` (the expected `LangChainDeprecationWarning` is left visible). In
  `transcripts/part2_task8_memory.txt`, "Should it be escalated?" resolves to `TKT-0003` from the earlier turn, while the same sentence in a fresh session asks for a ticket id.
- **Schema:** every crew response is a `SupportAnswer` (Pydantic, `output_pydantic=` on the Composer task) and is re-validated in code; an invalid payload is shown being rejected.
- **Guardrails:** phone-number masking (several formats; ticket ids and ordinary numbers are untouched), regex prompt-injection detection, and an output groundedness guard that refuses
  answers without single-sentence support in the retrieved context. Each is demonstrated firing in `transcripts/part2_task10_guardrails.txt`.
  Rider/driver name, address and payment details are free text and out of scope for the keyless masker (only fabricated examples are used).

## Part 3 - Evaluation, observability, deployment

- **Task 11:** `POST /ask`, `POST /add-document`, `GET /health` and `WS /ws/chat`. The WebSocket catches `WebSocketDisconnect`, drops the session, and the server keeps serving; the transcript
  shows a client dropping mid-conversation followed by a new client being served normally.
- **Task 12:** one JSON-Lines entry per request in `logs/requests.jsonl` with `trace_id` and `duration_ms`; everything is passed through the same masker the model sees before it is written, and
  the run asserts that no raw phone number appears anywhere in the file.
- **Task 13:** LLM-as-judge (prompt template in `transcripts/part3_task13_eval.txt`) under `MOCK_LLM`; 12 topic queries + a ticket lookup with a phone number + an out-of-scope query + a prompt-injection query.

{chr(10).join(ev_rows)}

## Part 4 - Resilience and governance

- **Task 14:** `RoundRobinGroupChat([Policy_Compliance_Reviewer, Final_Editor], max_turns=2, custom_message_types=[StructuredMessage[ReviewVerdict]])`, editor has `output_content_type=ReviewVerdict`
  (`approved`, `final_answer`, `reason`). Clean draft -> `approved={t14[0]['approved']}` unchanged; draft with an injected claim -> `approved={t14[1]['approved']}`, claim removed (`{t14[1]['reason'][:120]}...`).
- **Task 15 - application layer:** `make_agent()` calls `assert_tool_allowed(role, tool)` before building any agent, so only the Lookup Agent can hold `check_support_ticket_status`; wiring it to the Retrieval Agent
  or Composer raises `PermissionError` (shown in the transcript). **Risk class: {R['task15']['risk']}.** {__import__('governance').RISK_JUSTIFICATION}
- **Task 15 - runtime layer:** per-request budget of {config.MAX_REQUEST_TOKENS} tokens (${config.MAX_REQUEST_COST_USD:.4f}); a normal request used {R['task15']['normal_tokens']} tokens, a ~30k-character request is rejected up front, and a
  deliberately tight cap is enforced mid-request.
- **Task 16:** in-memory cache keyed by normalized query text. Repeated query: miss {t16['miss_ms']} ms (1 retrieval + 1 generation call) -> hit {t16['hit_ms']} ms
  (+{t16['hit_extra_retrieval']} retrieval, +{t16['hit_extra_generation']} generation calls), with different casing/punctuation hitting the same key.

## Notes on the MOCK_LLM implementation (the two silent pitfalls in the brief)

`MockCrewLLM` extends `crewai.llms.base_llm.BaseLLM`. (1) It never searches the whole conversation for `Observation:` (CrewAI's own system prompt contains that text); it only inspects assistant messages.
(2) It chooses a tool from the tool's declared argument schema (`record_id` vs `query`), never from its name. `tests/test_core.py` covers both.
"""
(config.BASE_DIR / "README.md").write_text(README)
print("README.md written" + (" (SMOKE-TEST banner included)" if smoke else ""))
