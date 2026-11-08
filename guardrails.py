"""Input-side (PII masking, prompt-injection) and output-side (groundedness) guardrails (Task 10)."""
import re

from embeddings import tokens
from mock_llm import FALLBACK, split_sentences
from schemas import SupportAnswer

# ------------------------------------------------------------------ PII masking
# Fixed-format PII for the Ola track: Indian mobile numbers, with/without +91 / 91 / 0 prefix and
# with spaces or dashes: 9876543210, +91 98765 43210, 98765-43210, 987-654-3210, 09876543210.
# Rider/driver name, address and payment details are free text: out of scope for a keyless masker.
_PHONE_CANDIDATE = re.compile(r"(?<![\w.])(?:\+?91[\s-]?|0)?[6-9](?:[\s-]?\d){9}(?![\w])")
PHONE_TOKEN = "[PHONE]"


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, [kinds of PII found])."""
    found: list[str] = []

    def _sub(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        elif digits.startswith("0") and len(digits) == 11:
            digits = digits[1:]
        if len(digits) == 10 and digits[0] in "6789":
            found.append("phone_number")
            return PHONE_TOKEN
        return m.group(0)

    return _PHONE_CANDIDATE.sub(_sub, text), found


def mask_obj(obj):
    """Apply the same masking to any nested str/list/dict (used before anything is logged)."""
    if isinstance(obj, str):
        return mask_pii(obj)[0]
    if isinstance(obj, list):
        return [mask_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: mask_obj(v) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------------ prompt-injection detection
_INJECTION_PATTERNS = {
    "ignore_previous_instructions": r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your)\b.{0,30}\b(instructions?|prompts?|rules?|guidelines?|policies|context)\b",
    "reveal_system_prompt": r"\b(reveal|show|print|repeat|leak|display)\b.{0,25}\b(system|hidden|initial|secret)\b.{0,15}\b(prompt|instructions?|message)\b",
    "role_hijack": r"\b(you are now|from now on you are|pretend (to be|you are)|act as if you have no)\b",
    "jailbreak_keywords": r"\b(jailbreak|developer mode|do anything now|\bDAN\b)\b",
    "bypass_safety": r"\b(bypass|disable|turn off)\b.{0,20}\b(safety|guardrails?|filters?|restrictions?)\b",
}


def detect_injection(text: str) -> list[str]:
    return [name for name, pat in _INJECTION_PATTERNS.items() if re.search(pat, text, re.I | re.S)]


# ------------------------------------------------------------------ output-side groundedness
SUPPORT_RATIO = 0.7  # a claim is supported if >=70% of its content tokens occur in ONE context sentence


def sentence_supported(sentence: str, context: str) -> bool:
    """A claim must be backed by a single context sentence.

    (Pooling words across the whole context is too lenient: "refunds are paid in cash within one hour"
    would pass because each word appears *somewhere* in the knowledge base.)
    """
    toks = tokens(sentence)
    if not toks:
        return True
    for ctx_sentence in split_sentences(context):
        ctx = set(tokens(ctx_sentence))
        if sum(t in ctx for t in toks) / len(toks) >= SUPPORT_RATIO:
            return True
    return False


def unsupported_sentences(answer: str, context: str) -> list[str]:
    return [s for s in split_sentences(answer) if not sentence_supported(s, context)]


def context_for(sources: list[str], ticket_statement: str | None) -> str:
    from rag import get_store  # lazy: include documents added at runtime via /add-document

    docs = get_store().docs
    parts = [docs[s]["text"] for s in sources if s in docs]
    if ticket_statement:
        parts.append(ticket_statement)
    return "\n".join(parts)


def groundedness_guard(answer: SupportAnswer, context: str) -> dict:
    """Refuse to answer when the retrieved context does not support the answer.

    Returns {"fired": bool, "reason": str|None, "answer": SupportAnswer}.
    """
    if answer.needs_clarification:  # asking the user for a missing ticket id makes no factual claim
        return {"fired": False, "reason": None, "answer": answer}
    if answer.ticket is None and not answer.sources:
        reason = "no retrieved context supports this question"
    elif not answer.grounded:
        reason = "crew reported the answer as ungrounded"
    else:
        bad = unsupported_sentences(answer.answer, context)
        reason = f"unsupported claim(s): {bad}" if bad else None
    if reason is None:
        return {"fired": False, "reason": None, "answer": answer}
    refused = answer.model_copy(update={"answer": FALLBACK, "sources": [], "ticket": None,
                                        "escalation_recommended": False, "grounded": False, "confidence": 0.0})
    return {"fired": True, "reason": reason, "answer": refused}
