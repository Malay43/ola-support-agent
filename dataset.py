"""Seeded, deterministic support-ticket dataset (Task 1).

Design choices (also copied into the README):
  * SEED = 13, N_RECORDS = 60, random.Random(SEED) is the only randomness source.
  * category weights / status weights are listed below.
  * resolution_time_hours: lognormal around a per-category median, clamped to
    [0.5, 96.0] h. Reasoning: a general inquiry or password reset is closed in
    minutes-to-hours, billing disputes in about a working day, and product
    defects (replacement/inspection) take days; 96 h is the P4 SLA ceiling.
    For tickets that are not resolved yet the value is the hours elapsed so far.
  * `escalated` is True for every ticket in status "Escalated" and otherwise
    for ~6% of tickets (a ticket can be flagged before status catches up).
No record is ever hand-edited: if a validation band fails, change SEED/weights.
"""
import math
import random
from collections import Counter

SEED = 13
N_RECORDS = 60

CATEGORIES = ["Billing", "Technical Issue", "Account Access", "Product Defect", "General Inquiry"]
CATEGORY_WEIGHTS = [0.26, 0.24, 0.18, 0.14, 0.18]

STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
STATUS_WEIGHTS = [0.18, 0.20, 0.14, 0.28, 0.20]

CATEGORY_MEDIAN_HOURS = {
    "Billing": 10.0,
    "Technical Issue": 24.0,
    "Account Access": 4.0,
    "Product Defect": 48.0,
    "General Inquiry": 2.0,
}
RESOLUTION_HOURS_RANGE = (0.5, 96.0)
BASE_ESCALATION_PROB = 0.06
ESCALATION_BAND = (10.0, 30.0)  # required % band


def generate_support_tickets(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    lo, hi = RESOLUTION_HOURS_RANGE
    records = []
    for i in range(1, n + 1):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        hours = rng.lognormvariate(math.log(CATEGORY_MEDIAN_HOURS[category]), 0.6)
        hours = round(min(max(hours, lo), hi), 1)
        days = rng.randint(0, 30)
        escalated = status == "Escalated" or rng.random() < BASE_ESCALATION_PROB
        records.append(
            {
                "record_id": f"TKT-{i:04d}",
                "category": category,
                "status": status,
                "resolution_time_hours": hours,
                "days_since_created": days,
                "escalated": bool(escalated),
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    cat = Counter(r["category"] for r in records)
    st = Counter(r["status"] for r in records)
    pct = 100.0 * sum(r["escalated"] for r in records) / len(records)
    return {"n": len(records), "per_category": dict(cat), "per_status": dict(st), "escalated_pct": round(pct, 1)}


def validate(records: list[dict]) -> None:
    s = summarize(records)
    assert s["n"] >= 40, "need >= 40 records"
    for c in CATEGORIES:
        assert s["per_category"].get(c, 0) >= 3, f"category {c} has < 3 records"
    for st in STATUSES:
        assert s["per_status"].get(st, 0) >= 1, f"status {st} missing"
    assert ESCALATION_BAND[0] <= s["escalated_pct"] <= ESCALATION_BAND[1], (
        f"escalated% {s['escalated_pct']} outside {ESCALATION_BAND}"
    )
    assert all(0 <= r["days_since_created"] <= 30 for r in records)


SUPPORT_TICKETS = generate_support_tickets()
validate(SUPPORT_TICKETS)
TICKET_INDEX = {r["record_id"]: r for r in SUPPORT_TICKETS}


def report() -> str:
    s = summarize(SUPPORT_TICKETS)
    lines = [f"SUPPORT_TICKETS: {s['n']} records (seed={SEED})", "Count per category:"]
    lines += [f"  {c:<17} {s['per_category'][c]}" for c in CATEGORIES]
    lines.append("Count per status:")
    lines += [f"  {c:<17} {s['per_status'][c]}" for c in STATUSES]
    lines.append(f"escalated=True: {s['escalated_pct']}%  (required band {ESCALATION_BAND[0]}-{ESCALATION_BAND[1]}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
    print("first 3 records:", SUPPORT_TICKETS[:3])
