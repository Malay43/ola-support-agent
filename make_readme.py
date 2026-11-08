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
    rows.append(f"| **Macro average** | | | **{e['macro_precision']:.3f}** | **{e['macro_recall']:.3f}** (F1 {e['macro_f1']:.3f}) |")
    return "\n".join(rows)


ev_rows = ["| No. | kind | accuracy | grounding | completeness | safety | question |", "|---|---|---|---|---|---|---|"]
for r in R["task13"]["rows"]:
    ev_rows.append(f"| {r['n']} | {r['kind']} | {r['accuracy']} | {r['grounding']} | {r['completeness']} | {r['safety']} | {r['question'][:80]} |")
a = R["task13"]["averages"]
ev_rows.append(f"| | **AVERAGE (15)** | **{a['accuracy']}** | **{a['grounding']}** | **{a['completeness']}** | **{a['safety']}** | |")

t7 = "\n".join(f"- \"{r['q']}\" - `rag_lookup` called {r['tool_calls_rag']}x, `check_support_ticket_status` called {r['tool_calls_lookup']}x" for r in R["task7"])
t14 = R["task14"]
t16 = R["task16"]

README = f"""# Ola Support Agent

This is my IITR capstone project for the Ola track (Support domain). It is a customer-support assistant that answers
questions about support policy from a small knowledge base, looks up the status of a ticket, and says "I don't know"
instead of guessing when a question is outside what it knows. Three CrewAI agents do the work, an Autogen review step
checks the answer before it goes out, and the whole thing is served through FastAPI.

{banner}Everything runs in `MOCK_LLM` mode by default, so you need no API keys and nothing is sent over the network. The only
download is a one-time fetch of the free embedding model, `{emb}`, which produced all the numbers in this README.

A few things worth knowing up front:

- Telemetry is switched off. `config.py` sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true`, along with
  `CREWAI_TRACING_ENABLED=false` and ChromaDB's `ANONYMIZED_TELEMETRY=False`. Every entry point imports `config.py` before
  `crewai` or `chromadb` loads, so `crew.kickoff()` never makes an outbound call. `run_all.py` prints these values when it starts.
- The knowledge-base policies are made up for this project. They are illustrative only and are not Ola's real internal policies.

## How to run

First, set up the environment. This creates a virtual environment and installs the pinned dependencies:

```bash
bash setup.sh
source venv/bin/activate
```

On Windows, run `.\\setup.ps1` and then `venv\\Scripts\\Activate.ps1` instead. During setup you will see a protobuf
dependency warning - that is expected, and the setup script explains why.

Then run every task's demo. It takes about a minute and saves a transcript for each task in `transcripts/`:

```bash
python run_all.py
```

To rebuild this README from the new results:

```bash
python make_readme.py
```

You can also start the API (`POST /ask`, `POST /add-document`, `WS /ws/chat`) and run the unit tests:

```bash
uvicorn app:app --port 8000
pytest -q
```

## What's in the repo

| File | What it does |
|---|---|
| `dataset.py` | Generates the seeded ticket dataset and checks it (Task 1) |
| `kb_docs.py` | The 12 knowledge-base documents (Task 2) |
| `embeddings.py`, `rag.py` | Embeddings, the two chunking strategies, the two ChromaDB collections, grounded answers, threshold calibration and precision/recall (Tasks 3-5) |
| `tools.py` | The `check_support_ticket_status` tool, the escalation score and the CrewAI tool wrappers (Task 6) |
| `mock_llm.py` | `MockCrewLLM`, the offline stand-in for a real model, plus the deterministic answer generator (Task 7) |
| `crew.py` | The Retrieval Agent, Lookup Agent and Response Composer, run with `.kickoff()` (Tasks 7 and 9) |
| `memory.py` | Conversation memory per session (Task 8) |
| `guardrails.py` | Phone-number masking, prompt-injection detection and the groundedness check (Task 10) |
| `app.py`, `logging_utils.py`, `pipeline.py` | The FastAPI app, request logging and the request pipeline (Tasks 11-12) |
| `evaluation.py` | LLM-as-judge evaluation on 15 questions (Task 13) |
| `review.py` | The Autogen review stage (Task 14) |
| `governance.py`, `cache.py` | Tool permissions, risk class, token budget and the response cache (Tasks 15-16) |
| `run_all.py`, `transcripts/` | Runs everything and keeps a transcript for every task |

## Part 1 - Dataset and RAG

**Task 1 - Designing the dataset**

The dataset is fully reproducible. `dataset.py` uses `random.Random(SEED)` as its only source of randomness, with
**seed {p1['seed']}** and **{ds['n']} tickets**. Here is how each field is generated:

- Categories are drawn with weights `{dict(zip(dataset.CATEGORIES, dataset.CATEGORY_WEIGHTS))}`.
- Statuses are drawn with weights `{dict(zip(dataset.STATUSES, dataset.STATUS_WEIGHTS))}`.
- Every ticket with status `Escalated` has `escalated = True`. Any other ticket is flagged with probability {dataset.BASE_ESCALATION_PROB}.
- Amount range (`resolution_time_hours`): drawn from a lognormal distribution around a median for each category,
  {dataset.CATEGORY_MEDIAN_HOURS}, and kept within {dataset.RESOLUTION_HOURS_RANGE} hours. Quick inquiries and account-access
  problems close within minutes to hours, billing takes about a working day, and product defects take days. 96 hours is the
  P4 SLA ceiling. For tickets that are still open, the value is the time elapsed so far.
- `days_since_created` is uniform between 0 and 30.
- I never edited a record by hand. I picked seed 13 from a sweep of seeds because it gave a balanced draw.

The result: per category `{ds['per_category']}`, per status `{ds['per_status']}`, and **{ds['escalated_pct']}%** of
tickets escalated, inside the required 10-30% band.

**Tasks 2 and 3 - Knowledge base and the two collections**

`kb_docs.py` holds 12 documents of 4-5 sentences each, one for each of the 12 required topics. I chunked them in two
ways: fixed-size chunks ({config.CHUNK_SIZE_CHARS} characters with {config.CHUNK_OVERLAP_CHARS} characters of overlap,
cut at word boundaries) and sentence-based chunks. Each set is embedded and stored in its own ChromaDB collection:
**{p1['chunk_counts']['fixed']} fixed chunks** and **{p1['chunk_counts']['sentence']} sentence chunks**.

**Task 4 - Grounded answers and the "I don't know" threshold**

Rather than picking a preset cut-off like 0.5 or 0.7, I measured it. I ran {len(cal['fixed']['in_scope'])} questions the
knowledge base can answer and {len(cal['fixed']['out_of_scope'])} it can't, recorded the top-1 cosine similarity for each,
and placed the threshold in the gap between the two groups.

For the fixed collection, the lowest in-scope score was {cal['fixed']['min_in_scope']:.3f} and the highest out-of-scope
score was {cal['fixed']['max_out_of_scope']:.3f}. The groups separate cleanly (`clusters_separate = {cal['fixed']['clusters_separate']}`),
so the threshold is **{cal['fixed']['threshold']}**.

{cal_table('fixed')}

For the sentence collection, the lowest in-scope score was {cal['sentence']['min_in_scope']:.3f} and the highest
out-of-scope score was {cal['sentence']['max_out_of_scope']:.3f}. These separate cleanly too
(`clusters_separate = {cal['sentence']['clusters_separate']}`), so the threshold is **{cal['sentence']['threshold']}**.

{cal_table('sentence')}

The deployed collection is **{rec}** (chosen in Task 5), so the live threshold is **{cal[rec]['threshold']}**. Six
answered questions and one out-of-scope question that falls back to "I don't know" are in `transcripts/part1_dataset_rag.txt`.

**Task 5 - Precision and recall for each collection**

I scored retrieval at the document level with top-k = 3. Chunks are mapped back to the document they came from, and
duplicates are removed before scoring.

Fixed collection:

{pr_table('fixed')}

Sentence collection:

{pr_table('sentence')}

{ev['recommendation_text']}

## Part 2 - CrewAI orchestration

**Task 6 - The ticket tool and the escalation score**

{R['task6']['justification']}

**Task 7 - The crew**

The crew has three agents. The Retrieval Agent holds `rag_lookup`, the Lookup Agent holds `check_support_ticket_status`,
and the Response Composer has no tools and writes the final answer. This is how often each tool was called for three
test questions:

{t7}

**Tasks 8 to 10 - Memory, structured output and guardrails**

- Memory: each session keeps its own history (`RunnableWithMessageHistory` over `InMemoryChatMessageHistory`; the expected
  `LangChainDeprecationWarning` is left visible). In `transcripts/part2_task8_memory.txt`, asking "Should it be escalated?"
  after a question about `TKT-0003` correctly refers to that ticket, while the same question in a new session asks which ticket you mean.
- Structured output: every crew answer is a `SupportAnswer` Pydantic model (set with `output_pydantic=` on the Composer's
  task) and is validated again in code. The transcript also shows an invalid answer being rejected.
- Guardrails: phone numbers are masked in several formats (ticket ids and ordinary numbers are left alone), prompt-injection
  attempts are caught with regex patterns, and a groundedness check refuses any answer that isn't backed by a single sentence
  in the retrieved context. Each one is shown firing in `transcripts/part2_task10_guardrails.txt`. Names, addresses and payment
  details are free text, so the masker doesn't attempt them; only made-up examples are used in this project.

## Part 3 - Evaluation, logging and deployment

- Task 11: the API has `POST /ask`, `POST /add-document`, `GET /health` and `WS /ws/chat`. If a WebSocket client disconnects,
  the server catches `WebSocketDisconnect`, clears that session and carries on. The transcript shows one client dropping
  mid-conversation and a new client being served normally afterwards.
- Task 12: every request is logged as one line in `logs/requests.jsonl`, with a `trace_id` and `duration_ms`. Each entry goes
  through the same masker as the model input before it is written, and the run checks that no raw phone number appears anywhere in the file.
- Task 13: an LLM-as-judge scores 15 questions under `MOCK_LLM` (the prompt template is in `transcripts/part3_task13_eval.txt`):
  12 policy topics, a ticket lookup that includes a phone number, an out-of-scope question and a prompt-injection attempt.

{chr(10).join(ev_rows)}

## Part 4 - Resilience and governance

- Task 14 - Review stage: an Autogen `RoundRobinGroupChat` with two agents, a Policy Compliance Reviewer and a Final Editor
  (`max_turns=2`, `custom_message_types=[StructuredMessage[ReviewVerdict]]`). The editor returns a `ReviewVerdict` with
  `approved`, `final_answer` and `reason`. A clean draft passes unchanged (`approved={t14[0]['approved']}`). A draft with an
  invented claim is caught (`approved={t14[1]['approved']}`) and the claim is removed: "{t14[1]['reason']}"
- Task 15 - Tool permissions: `make_agent()` calls `assert_tool_allowed(role, tool)` before it builds any agent, so only the
  Lookup Agent can hold `check_support_ticket_status`. Giving it to the Retrieval Agent or the Composer raises a
  `PermissionError`, which the transcript shows.
- Task 15 - Risk class: **{R['task15']['risk']}**. {__import__('governance').RISK_JUSTIFICATION}
- Task 15 - Budget: each request may use up to {config.MAX_REQUEST_TOKENS} tokens (${config.MAX_REQUEST_COST_USD:.4f}). A normal
  request used {R['task15']['normal_tokens']} tokens. A request of about 30,000 characters is rejected before any work starts,
  and a deliberately tight cap is enforced partway through a request.
- Task 16 - Cache: answers are cached by the normalized question text. The first call took {t16['miss_ms']} ms with one
  retrieval and one generation call. Repeating it took {t16['hit_ms']} ms with {t16['hit_extra_retrieval']} retrieval and
  {t16['hit_extra_generation']} generation calls, even with different capitalisation and punctuation.

## How the mock LLM avoids the two pitfalls in the brief

`MockCrewLLM` extends `crewai.llms.base_llm.BaseLLM`, CrewAI's documented way to plug in a custom model.

1. It never searches the whole conversation for `Observation:`, because CrewAI's own system prompt contains that word. It
   only looks at the assistant's messages.
2. It picks a tool by the arguments the tool declares (`record_id` or `query`), never by the tool's name.

`tests/test_core.py` has tests for both.
"""
(config.BASE_DIR / "README.md").write_text(README)
print("README.md written" + (" (SMOKE-TEST banner included)" if smoke else ""))
