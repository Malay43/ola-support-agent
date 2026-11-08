"""Deterministic MOCK_LLM pieces that need no API key and no network.

Part 1 (this section): the grounded-answer generator + judge used by the
evaluation. The CrewAI `BaseLLM` subclass and the Autogen client are appended
further down in this file.
"""
import json
import re

import config
from cache import COUNTERS
from embeddings import _stem, tokens

FALLBACK = "I don't know - I couldn't find that in the support knowledge base."


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _complete(sentence: str) -> bool:
    return bool(re.match(r"[A-Z0-9]", sentence)) and sentence.endswith((".", "!", "?"))


def generate_grounded(question: str, contexts: list[str], max_sentences: int = 2) -> str:
    """Extractive 'generation' using ONLY the retrieved context sentences."""
    COUNTERS["generation_calls"] += 1
    q = set(tokens(question))
    cands, seen = [], set()
    for ctx in contexts:
        for s in split_sentences(ctx):
            if _complete(s) and s not in seen:
                seen.add(s)
                cands.append(s)
    if not cands:
        return " ".join(contexts[:1]).strip() or FALLBACK
    def overlap(sent: str) -> int:
        return len(q & set(tokens(sent)))

    ranked = sorted(enumerate(cands), key=lambda p: (-overlap(p[1]), len(p[1]), p[0]))
    best_ov = overlap(ranked[0][1])
    keep = [ranked[0]]
    for p in ranked[1:max_sentences]:
        if overlap(p[1]) >= max(2, -(-best_ov * 6 // 10)):  # only add a 2nd sentence if it is really on-topic
            keep.append(p)
    return " ".join(s for _, s in sorted(keep, key=lambda p: p[0]))


# =====================================================================================
# CrewAI MOCK_LLM  (extends crewai.llms.base_llm.BaseLLM - CrewAI's documented extension point)
# =====================================================================================
TICKET_RE = re.compile(r"\bTKT-\d{4}\b", re.I)
_LOOKUP_NOISE = {_stem(w) for w in "status ticket check tell show look lookup give find need want please escalate escalated escalation know phone mobile number contact call reach".split()}


def clean_question(q: str) -> str:
    """Question text without ticket ids / masked PII markers, used for the RAG lookup."""
    q = TICKET_RE.sub(" ", q)
    q = re.sub(r"\[[A-Z_]+(?:\d*)\]|\b[Xx]{6}\d{4}\b", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def is_pure_ticket_lookup(q: str) -> bool:
    """True when, once ids and filler words are removed, no policy topic is left to search for."""
    content = [t for t in tokens(clean_question(q)) if t not in _LOOKUP_NOISE]
    return len(content) < 2


def parse_tools_from_system(system_text: str) -> list[dict]:
    """Read the tool list from CrewAI's ReAct system prompt: [{'name':..., 'args': [...]}]."""
    tools = []
    for m in re.finditer(r"Tool Name:\s*(\S+)\s*\nTool Arguments:\s*(\{.*?\n\})\s*\nTool Description:", system_text, re.S):
        try:
            props = json.loads(m.group(2)).get("properties", {})
        except json.JSONDecodeError:
            props = {}
        tools.append({"name": m.group(1), "args": list(props)})
    return tools


def _extract_question(text: str) -> str:
    m = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else text.strip()[:400]


def compose_structured_answer(question: str, retrieval_text: str, lookup_text: str) -> dict:
    """Deterministic Response-Composer logic: merge policy answer + ticket record into SupportAnswer JSON."""
    parts, sources, grounded, conf = [], [], False, 0.0
    ticket, esc_reco, clarify = None, False, False

    lm = re.search(r"LOOKUP_RESULT:\s*(\{.*\})", lookup_text, re.S)
    if lm:
        try:
            rec = json.loads(lm.group(1))
        except json.JSONDecodeError:
            rec = {}
        if rec.get("found"):
            ticket = {
                "record_id": rec["record_id"], "status": rec["status"],
                "resolution_time_hours": rec["resolution_time_hours"],
                "escalation_score": rec["escalation_score"], "recommend_escalation": rec["recommend_escalation"],
            }
            esc_reco = bool(rec["recommend_escalation"])
            parts.append(ticket_statement(rec))
            grounded, conf = True, 1.0
        elif rec.get("error"):
            parts.append(f"{rec['error']}. Please check the ticket id.")

    rm = re.search(r"GROUNDED=(\w+) TOP_SCORE=([\d.]+) SOURCES=([^\n]*)\nANSWER:\s*(.+)", retrieval_text, re.S)
    if rm and rm.group(1) == "True":
        parts.append(rm.group(4).strip())
        sources = [s for s in rm.group(3).split(",") if s and s != "none"]
        grounded = True
        conf = max(conf, round(min(1.0, float(rm.group(2)) * 1.5), 2))
    elif rm and not parts:
        parts.append(FALLBACK)

    if not parts:
        if "NO_TICKET_ID" in lookup_text and "RAG_SKIPPED" in retrieval_text:
            parts.append("Please share the ticket id (for example TKT-0007) so I can look it up.")
            grounded, conf, clarify = True, 0.5, True
        else:
            parts.append(FALLBACK)
    return {
        "answer": " ".join(parts), "sources": sources, "ticket": ticket,
        "escalation_recommended": esc_reco, "grounded": grounded, "needs_clarification": clarify,
        "confidence": conf,
    }


try:
    from crewai.llms.base_llm import BaseLLM

    class MockCrewLLM(BaseLLM):
        """Deterministic ReAct-speaking model. No API key, no network.

        Pitfall 1 (brief): CrewAI's system prompt contains the literal text
        'Observation: the result of the action'. We therefore NEVER search the whole conversation
        for 'Observation:' - we only look inside the *assistant* messages (the model's own text,
        into which CrewAI appends the real tool observation).
        Pitfall 2 (brief): we never pick a tool by matching its *name*; we dispatch on the tool's
        declared argument schema (`record_id` -> ticket lookup, `query` -> retrieval).
        """

        calls: int = 0

        def __init__(self, **kw):
            super().__init__(model="mock-llm", **kw)

        def supports_function_calling(self) -> bool:
            return False  # force CrewAI's text ReAct protocol

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 16000

        # -- helpers
        @staticmethod
        def _norm(messages):
            if isinstance(messages, str):
                return [{"role": "user", "content": messages}]
            return [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else str(m["content"])} for m in messages]

        @staticmethod
        def _last_observation(msgs) -> str | None:
            for m in reversed(msgs):
                if m["role"] == "assistant" and "\nObservation:" in m["content"]:
                    return m["content"].split("\nObservation:", 1)[1].strip()
            return None

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            import governance

            self.calls += 1
            msgs = self._norm(messages)
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")
            first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            agent_tools = parse_tools_from_system(system)
            obs = self._last_observation(msgs)
            question = _extract_question(first_user)

            if not agent_tools:  # ---- Response Composer: no tools, builds the structured answer
                ctx = first_user.split("This is the context you're working with:", 1)[-1]
                ctx = ctx.split("\n\nProvide your complete response", 1)[0]  # CrewAI's no-tools template suffix
                parts = [p.strip() for p in ctx.split("----------")]  # CrewAI joins task outputs with this separator
                r_txt = next((p for p in parts if p.startswith(("RAG_RESULT:", "RAG_SKIPPED"))), "")
                l_txt = next((p for p in parts if p.startswith(("LOOKUP_RESULT:", "NO_TICKET_ID"))), "")
                payload = compose_structured_answer(question, r_txt.replace("RAG_RESULT:", "").strip(), l_txt)
                out = "Thought: I now know the final answer\nFinal Answer: " + json.dumps(payload)
            elif obs is not None:  # ---- a tool already ran: wrap its observation as the final answer
                label = "LOOKUP_RESULT" if any("record_id" in t["args"] for t in agent_tools) else "RAG_RESULT"
                out = f"Thought: I now know the final answer\nFinal Answer: {label}: {obs}"
            else:  # ---- first step: choose a tool by its declared argument schema
                tool = next((t for t in agent_tools if "record_id" in t["args"]), None)
                if tool is not None:
                    m = TICKET_RE.search(question)
                    if m:
                        out = (f"Thought: The question names a ticket, so I will look it up.\nAction: {tool['name']}\n"
                               f"Action Input: {json.dumps({'record_id': m.group(0).upper()})}")
                    else:
                        out = "Thought: No ticket id was given.\nFinal Answer: NO_TICKET_ID: no ticket id in the question"
                else:
                    tool = next((t for t in agent_tools if "query" in t["args"]), agent_tools[0])
                    if is_pure_ticket_lookup(question):
                        out = "Thought: This is only a ticket lookup.\nFinal Answer: RAG_SKIPPED: no policy topic in the question"
                    else:
                        out = (f"Thought: I should search the policy knowledge base.\nAction: {tool['name']}\n"
                               f"Action Input: {json.dumps({'query': clean_question(question)})}")

            meter = governance.current_meter()
            if meter is not None:
                meter.add(governance.estimate_tokens(" ".join(m["content"] for m in msgs)) + governance.estimate_tokens(out))
            return out

except Exception:  # pragma: no cover - crewai not installed
    MockCrewLLM = None


def ticket_statement(rec: dict) -> str:
    """Canonical sentence(s) describing a ticket record. Used by the composer AND as review context."""
    verdict = "recommend escalating it" if rec["recommend_escalation"] else "no escalation is needed right now"
    return (
        f"Ticket {rec['record_id']} is currently {rec['status']} (resolution time {rec['resolution_time_hours']} hours). "
        f"Its escalation score is {rec['escalation_score']} (threshold {rec['threshold']}), so {verdict}."
    )
