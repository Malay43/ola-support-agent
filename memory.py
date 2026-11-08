"""Session memory with LangChain's RunnableWithMessageHistory (Task 8).

NOTE: RunnableWithMessageHistory emits a LangChainDeprecationWarning pointing at LangGraph
persistence. This is expected (the brief says so) and the class works correctly; we do not silence it.
"""
import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from mock_llm import TICKET_RE

_SESSIONS: dict[str, InMemoryChatMessageHistory] = {}
_REFERS_BACK = re.compile(r"\b(it|its|that ticket|this ticket|the same ticket|that one|same one|the ticket)\b", re.I)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return _SESSIONS.setdefault(session_id, InMemoryChatMessageHistory())


def clear_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


def last_ticket_in(history) -> str | None:
    for msg in reversed(history):
        found = TICKET_RE.findall(msg.content if isinstance(msg.content, str) else "")
        if found:
            return found[-1].upper()
    return None


def resolve_follow_up(question: str, history) -> tuple[str, str | None]:
    """If the question refers back to 'it / that ticket' and names no ticket, inject the last ticket id."""
    if TICKET_RE.search(question) or not _REFERS_BACK.search(question):
        return question, None
    tid = last_ticket_in(history)
    if tid is None:
        return question, None
    return f"{question} (this is about ticket {tid})", tid


def build_chat(core_fn):
    """core_fn(resolved_question) -> dict payload. Returns a session-aware runnable."""

    def _run(inputs: dict) -> AIMessage:
        resolved, carried = resolve_follow_up(inputs["input"], inputs.get("history", []))
        payload = core_fn(resolved)
        payload["memory_resolved_ticket"] = carried
        return AIMessage(content=payload["answer_text"], additional_kwargs={"payload": payload})

    return RunnableWithMessageHistory(
        RunnableLambda(_run), get_session_history, input_messages_key="input", history_messages_key="history"
    )
