from __future__ import annotations

"""FastAPI entrypoint for chatbot endpoints."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.auth import CurrentUser, require_admin, require_user
from app.response import created, failure, success
from app.schemas import ChatRequest, IngestRequest
from app.service import assistant_service
from app.sync_scheduler import sync_scheduler


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start/stop background sync scheduler with app lifecycle."""

    await sync_scheduler.start()
    try:
        yield
    finally:
        await sync_scheduler.stop()

app = FastAPI(
    title="TechXchange Chatbot API",
    version="0.2.0",
    summary="Python assistant API for the TechXchange chatbot",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Convert FastAPI HTTP errors to the project envelope format."""

    return JSONResponse(status_code=exc.status_code, content=failure(exc.status_code, str(exc.detail)))


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected errors and return a consistent 500 response body."""

    return JSONResponse(status_code=500, content=failure(500, str(exc)))


@app.get("/health")
async def healthcheck() -> dict:
    """Lightweight process health endpoint."""

    return {"status": "ok"}


@app.get("/api/assistant/health")
async def assistant_health() -> dict:
    """Return assistant readiness, provider flags, and storage diagnostics."""

    return success("Assistant service is healthy", assistant_service.health())


@app.post("/api/assistant/chat")
async def chat(req: ChatRequest, user: CurrentUser = Depends(require_user)) -> dict:
    """Main chat endpoint for RAG responses."""

    try:
        result = await assistant_service.chat(
            user_id=user.user_id,
            message=req.message,
            locale=req.locale,
            conversation_id=req.conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return success("Assistant response", result)


@app.get("/api/assistant/conversations")
async def conversations(
    limit: int = 20,
    user: CurrentUser = Depends(require_user),
) -> dict:
    """List conversation headers for current user."""

    rows = assistant_service.list_conversations(user.user_id, limit)
    return success("Conversations fetched", {"conversations": rows})


@app.get("/api/assistant/messages/{conversation_id}")
async def messages(
    conversation_id: int,
    limit: int = 50,
    user: CurrentUser = Depends(require_user),
) -> dict:
    """Fetch message history for one conversation."""

    try:
        rows = assistant_service.get_messages(user.user_id, conversation_id, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return success("Messages fetched", {"messages": rows})


@app.post("/api/assistant/ingest")
async def ingest(
    req: IngestRequest,
    _: CurrentUser = Depends(require_admin),
) -> dict:
    """Admin endpoint to ingest manual knowledge documents."""

    try:
        result = await assistant_service.ingest_documents(
            [item.model_dump() for item in req.documents]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return created("Ingestion completed", result)


@app.post("/api/assistant/reindex")
async def reindex(
    _: CurrentUser = Depends(require_admin),
) -> dict:
    """Admin endpoint to sync latest product data into chatbot KB."""

    try:
        result = await assistant_service.reindex()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return success("Reindex completed", result)
