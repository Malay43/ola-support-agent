"""Pydantic schemas shared by the crew, API and review stage."""
from typing import Optional

from pydantic import BaseModel, Field


class TicketInfo(BaseModel):
    record_id: str
    status: str
    resolution_time_hours: float
    escalation_score: float = Field(ge=0.0, le=1.0)
    recommend_escalation: bool


class SupportAnswer(BaseModel):
    """Every crew response must conform to this model (Task 9)."""

    answer: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list, description="KB document ids used")
    ticket: Optional[TicketInfo] = None
    escalation_recommended: bool = False
    grounded: bool = True
    needs_clarification: bool = False  # True when the reply asks the user for missing info (e.g. a ticket id)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReviewVerdict(BaseModel):
    """Verdict produced by the Autogen Final-Editor (Task 14)."""

    approved: bool
    final_answer: str
    reason: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=200000)
    session_id: str = "default"


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    blocked: bool = False
    block_reason: Optional[str] = None
    sources: list[str] = []
    ticket: Optional[TicketInfo] = None
    escalation_recommended: bool = False
    grounded: bool = True
    review_approved: Optional[bool] = None
    review_reason: Optional[str] = None
    pii_masked: list[str] = []
    cache_hit: bool = False
    duration_ms: float = 0.0


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=20)
    topic: str = "user-added"


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_indexed: int
