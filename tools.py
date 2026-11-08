"""The two tools (Tasks 3-6) as plain functions + CrewAI BaseTool wrappers."""
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import dataset
from cache import COUNTERS
from rag import grounded_answer

# escalation_score = 0.6 * escalated + 0.4 * (days_since_created / 30)
#   * `escalated` (0/1) is the dominant signal: a ticket already flagged is far more urgent.
#   * days/30 is the normalised age in [0,1]: an older ticket has had more time to breach SLA.
#   * weights sum to 1, so the score is always in [0, 1].
W_ESCALATED, W_AGE = 0.6, 0.4


def escalation_score(record: dict) -> float:
    age = record["days_since_created"] / 30.0
    return round(W_ESCALATED * float(record["escalated"]) + W_AGE * age, 4)


_ALL_SCORES = [escalation_score(r) for r in dataset.SUPPORT_TICKETS]
ESCALATION_PERCENTILE = 80
ESCALATION_THRESHOLD = round(float(np.percentile(_ALL_SCORES, ESCALATION_PERCENTILE)), 2)


def escalation_justification() -> str:
    import math

    n_above = sum(sc >= ESCALATION_THRESHOLD for sc in _ALL_SCORES)
    esc_ok = "every ticket already flagged `escalated` (score >= 0.6) is recommended for escalation"
    if ESCALATION_THRESHOLD > W_ESCALATED:
        d = math.ceil((ESCALATION_THRESHOLD - W_ESCALATED) / W_AGE * 30)
        esc_ok = f"a flagged ticket is recommended once it is about {d} days old"
    non_esc_days = math.ceil(ESCALATION_THRESHOLD / W_AGE * 30)
    non_esc = (f"an unflagged ticket is recommended only when it is at least {non_esc_days} days old"
               if non_esc_days <= 30 else "an unflagged ticket can never reach it (maximum score "
               f"{W_AGE})")
    return (
        f"escalation_score = {W_ESCALATED} * escalated + {W_AGE} * (days_since_created / 30), always in [0, 1]. "
        f"Recommend escalation when score >= {ESCALATION_THRESHOLD}: this is the {ESCALATION_PERCENTILE}th percentile of "
        f"the scores over my {len(_ALL_SCORES)} generated tickets, so {n_above} of {len(_ALL_SCORES)} tickets "
        f"({100 * n_above / len(_ALL_SCORES):.0f}%) are above it, i.e. only the top fifth of the queue is escalated. "
        f"In practice {esc_ok}, and {non_esc}."
    )


def check_support_ticket_status(record_id: str) -> dict:
    """Look up one ticket and return status, resolution time and a designed escalation score."""
    COUNTERS["tool_calls_lookup"] += 1
    rid = record_id.strip().upper()
    rec = dataset.TICKET_INDEX.get(rid)
    if rec is None:
        return {"found": False, "record_id": rid, "error": f"No ticket with id {rid}"}
    score = escalation_score(rec)
    return {
        "found": True,
        "record_id": rid,
        "status": rec["status"],
        "resolution_time_hours": rec["resolution_time_hours"],
        "escalation_score": score,
        "recommend_escalation": score >= ESCALATION_THRESHOLD,
        "threshold": ESCALATION_THRESHOLD,
    }


class _RagArgs(BaseModel):
    query: str = Field(..., description="The support-policy question to look up in the knowledge base")


class _TicketArgs(BaseModel):
    record_id: str = Field(..., description="Ticket id such as TKT-0007")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Search the support-policy knowledge base and return a grounded answer with source ids."
    args_schema: type[BaseModel] = _RagArgs

    def _run(self, query: str) -> str:
        COUNTERS["tool_calls_rag"] += 1
        r = grounded_answer(query)
        return (
            f"GROUNDED={r['grounded']} TOP_SCORE={r['top_score']:.3f} SOURCES={','.join(r['sources']) or 'none'}\n"
            f"ANSWER: {r['answer']}"
        )


class TicketStatusTool(BaseTool):
    name: str = "check_support_ticket_status"
    description: str = (
        "Look up a support ticket by id and return its status, resolution time and escalation score."
    )
    args_schema: type[BaseModel] = _TicketArgs

    def _run(self, record_id: str) -> str:
        import json

        return json.dumps(check_support_ticket_status(record_id))
