"""FastAPI deployment (Task 11): 2 HTTP endpoints + 1 WebSocket that survives client disconnects."""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

import config  # noqa: F401  (sets telemetry-off env vars first)
from logging_utils import log_request, new_trace_id
from pipeline import answer_question
from rag import get_store
from schemas import AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse

@asynccontextmanager
async def lifespan(_: FastAPI):
    get_store()  # build + calibrate the knowledge base once, not on the first request
    yield


app = FastAPI(title="Ola Support Agent (CrewAI + Autogen review)", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    s = get_store()
    return {"status": "ok", "mock_llm": config.MOCK_LLM, "embedder": s.embedder.name,
            "deployed_collection": s.recommended, "documents": len(s.docs)}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    result = await run_in_threadpool(answer_question, req.question, req.session_id, "/ask")
    result.pop("_debug", None)
    return AskResponse(**result)


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    trace_id, t0 = new_trace_id(), time.perf_counter()
    out = await run_in_threadpool(get_store().add_document, req.doc_id, req.text, req.topic)
    log_request({"trace_id": trace_id, "endpoint": "/add-document", "doc_id": req.doc_id, "outcome": "indexed",
                 "chunks_indexed": out["chunks_indexed"], "duration_ms": round((time.perf_counter() - t0) * 1000, 2)})
    return AddDocumentResponse(trace_id=trace_id, **out)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    """Multi-turn chat. One session id per connection, so memory persists across messages on this socket only."""
    await ws.accept()
    session_id = f"ws-{uuid.uuid4().hex[:8]}"
    try:
        while True:
            text = await ws.receive_text()
            try:
                result = await run_in_threadpool(answer_question, text, session_id, "/ws/chat")
                result.pop("_debug", None)
                await ws.send_json(AskResponse(**result).model_dump())
            except (WebSocketDisconnect, RuntimeError):
                raise
            except Exception as e:  # one bad message must not kill the connection or the server
                log_request({"trace_id": new_trace_id(), "endpoint": "/ws/chat", "session_id": session_id,
                             "outcome": "error", "error": type(e).__name__})
                await ws.send_json({"error": "internal error", "detail": type(e).__name__})
    except (WebSocketDisconnect, RuntimeError):
        log_request({"trace_id": new_trace_id(), "endpoint": "/ws/chat", "session_id": session_id,
                     "outcome": "client_disconnected"})
        from memory import clear_session
        clear_session(session_id)
