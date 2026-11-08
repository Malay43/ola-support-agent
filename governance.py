"""Four-layer AI governance helpers (Task 15): least-autonomy guard, risk class, runtime budget."""
import contextvars

import config

# ------------------------------------------------------------------ Application layer
# Least autonomy: each tool has exactly one role allowed to hold it.
TOOL_POLICY: dict[str, set[str]] = {
    "check_support_ticket_status": {"Lookup Agent"},
    "rag_lookup": {"Retrieval Agent"},
}


def assert_tool_allowed(role: str, tool_name: str) -> None:
    allowed = TOOL_POLICY.get(tool_name)
    if allowed is not None and role not in allowed:
        raise PermissionError(
            f"Least-autonomy violation: agent '{role}' may not hold tool '{tool_name}' "
            f"(allowed: {sorted(allowed)})."
        )


RISK_LEVEL = "Medium"
RISK_JUSTIFICATION = (
    "I classify this system as MEDIUM risk. It is a customer-support assistant that answers policy questions and "
    "reads ticket status, which the scheme lists under 'customer support tickets' (Medium), not under summarisation "
    "(Low). It is not High: it handles no medical data, makes no hiring decision and touches no financial account - "
    "it cannot issue refunds or change a ticket, it only reads one record and quotes policy. The residual risks are a "
    "wrong or ungrounded answer and leakage of a phone number, and both are controlled by the retrieval-similarity "
    "groundedness gate, the Autogen review stage, input-side PII masking (also applied before logging), prompt-injection "
    "detection, a per-request token budget, and a read-only tool that only the Lookup Agent may hold."
)

# ------------------------------------------------------------------ Runtime layer
class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic estimate (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


class RequestMeter:
    """Tracks tokens and cost for ONE request and refuses to go over the cap."""

    def __init__(self, max_tokens: int = config.MAX_REQUEST_TOKENS):
        self.max_tokens = max_tokens
        self.tokens = 0

    @property
    def cost_usd(self) -> float:
        return round(self.tokens / 1000 * config.COST_PER_1K_TOKENS_USD, 6)

    def check_input(self, text: str) -> None:
        est = estimate_tokens(text)
        if est > self.max_tokens:
            raise BudgetExceeded(
                f"Request rejected: ~{est} input tokens exceeds the per-request budget of {self.max_tokens} tokens "
                f"(cap ${config.MAX_REQUEST_COST_USD:.4f})."
            )

    def add(self, tokens: int) -> None:
        self.tokens += tokens
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(
                f"Budget exceeded mid-request: {self.tokens} tokens used > cap {self.max_tokens}."
            )


_METER: contextvars.ContextVar[RequestMeter | None] = contextvars.ContextVar("request_meter", default=None)


def current_meter() -> RequestMeter | None:
    return _METER.get()


def set_meter(m: RequestMeter | None):
    return _METER.set(m)


def reset_meter(token) -> None:
    _METER.reset(token)
