"""End-to-end request pipeline:

 budget check -> PII mask -> injection check -> (session memory) CrewAI crew -> groundedness guard
 -> Autogen review -> structured JSONL log.
"""
import time

import governance
from cache import COUNTERS
from crew import run_crew
from guardrails import context_for, detect_injection, groundedness_guard, mask_pii
from logging_utils import log_request, new_trace_id
from memory import build_chat
from mock_llm import FALLBACK, ticket_statement
from review import review_answer
from schemas import AskResponse, SupportAnswer, TicketInfo


def _core(resolved_question: str) -> dict:
    out = run_crew(resolved_question)
    sa: SupportAnswer = out["answer"]
    return {"answer_text": sa.answer, "support_answer": sa, "task_outputs": out["task_outputs"],
            "resolved_question": resolved_question}


_CHAT = build_chat(_core)


def _ticket_context(sa: SupportAnswer) -> str | None:
    if sa.ticket is None:
        return None
    from tools import ESCALATION_THRESHOLD
    rec = sa.ticket.model_dump()
    rec["threshold"] = ESCALATION_THRESHOLD
    return ticket_statement(rec)


def _blocked(trace_id, reason, message, t0, masked, pii, endpoint, session_id, extra=None) -> dict:
    resp = AskResponse(trace_id=trace_id, answer=message, blocked=True, block_reason=reason, grounded=False,
                       pii_masked=pii, duration_ms=round((time.perf_counter() - t0) * 1000, 2)).model_dump()
    log_request({"trace_id": trace_id, "endpoint": endpoint, "session_id": session_id, "question": masked,
                 "outcome": "blocked", "block_reason": reason, "pii_masked": pii,
                 "duration_ms": resp["duration_ms"], **(extra or {})})
    return resp


def answer_question(raw_question: str, session_id: str = "default", endpoint: str = "/ask",
                    review: bool = True, trace_id: str | None = None, max_tokens: int | None = None) -> dict:
    trace_id = trace_id or new_trace_id()
    t0 = time.perf_counter()
    meter = governance.RequestMeter(max_tokens) if max_tokens else governance.RequestMeter()
    tok = governance.set_meter(meter)
    hits_before = COUNTERS["cache_hits"]
    masked, pii = mask_pii(raw_question)  # mask FIRST so even a rejected request is logged without raw PII
    try:
        try:
            meter.check_input(raw_question)
        except governance.BudgetExceeded as e:
            return _blocked(trace_id, "budget_exceeded", str(e), t0, masked[:200] + " ...[truncated for log]", pii,
                            endpoint, session_id, {"tokens": meter.tokens})
        inj = detect_injection(masked)
        if inj:
            return _blocked(trace_id, "prompt_injection",
                            "I can't help with that request - it looks like an attempt to override my instructions.",
                            t0, masked, pii, endpoint, session_id, {"injection_patterns": inj})

        try:
            msg = _CHAT.invoke({"input": masked}, config={"configurable": {"session_id": session_id}})
        except governance.BudgetExceeded as e:
            return _blocked(trace_id, "budget_exceeded", str(e), t0, masked, pii, endpoint, session_id,
                            {"tokens": meter.tokens})
        payload = msg.additional_kwargs["payload"]
        sa: SupportAnswer = payload["support_answer"]

        ctx = context_for(sa.sources, _ticket_context(sa))
        guard = groundedness_guard(sa, ctx)
        sa = guard["answer"]
        verdict = None
        if review and not guard["fired"] and not sa.needs_clarification:  # a clarifying question makes no claim to review
            try:
                verdict, _ = review_answer(sa.answer, ctx)
            except governance.BudgetExceeded as e:
                return _blocked(trace_id, "budget_exceeded", str(e), t0, masked, pii, endpoint, session_id,
                                {"tokens": meter.tokens})
            if not verdict.approved:
                sa = sa.model_copy(update={"answer": verdict.final_answer})
        ticket = TicketInfo(**sa.ticket.model_dump()) if sa.ticket else None
        cache_hit = COUNTERS["cache_hits"] > hits_before
        resp = AskResponse(
            trace_id=trace_id, answer=sa.answer, sources=sa.sources, ticket=ticket,
            escalation_recommended=sa.escalation_recommended, grounded=sa.grounded,
            review_approved=verdict.approved if verdict else None, review_reason=verdict.reason if verdict else None,
            pii_masked=pii, cache_hit=cache_hit, duration_ms=round((time.perf_counter() - t0) * 1000, 2)).model_dump()
        resp["_debug"] = {"guard_fired": guard["fired"], "guard_reason": guard["reason"],
                          "memory_resolved_ticket": payload.get("memory_resolved_ticket"),
                          "resolved_question": payload["resolved_question"], "task_outputs": payload["task_outputs"],
                          "tokens": meter.tokens, "cost_usd": meter.cost_usd}
        log_request({"trace_id": trace_id, "endpoint": endpoint, "session_id": session_id, "question": masked,
                     "outcome": "answered", "grounded": sa.grounded, "sources": sa.sources,
                     "guard_fired": guard["fired"], "review_approved": resp["review_approved"], "pii_masked": pii,
                     "cache_hit": cache_hit, "tokens": meter.tokens, "cost_usd": meter.cost_usd,
                     "duration_ms": resp["duration_ms"]})
        return resp
    finally:
        governance.reset_meter(tok)
