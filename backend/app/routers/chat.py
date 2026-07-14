"""Chat session and messaging API endpoints with SSE streaming."""

import json
import logging
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.chat import (
    create_session,
    get_session,
    delete_session,
    get_session_history,
    get_sessions_for_document,
    list_all_sessions,
    process_message,
    process_message_stream,
)
from app.models.schemas import (
    ChatSession,
    ChatSessionCreate,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatHistoryResponse,
    ChatMessage,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(body: ChatSessionCreate = None):
    """Create a new chat session, optionally scoped to a document."""
    if body is None:
        body = ChatSessionCreate()
    
    session = create_session(
        document_id=body.document_id,
        document_ids=body.document_ids,
    )
    return ChatSession(
        session_id=session["session_id"],
        created_at=session["created_at"],
        message_count=0,
        document_id=session.get("document_id"),
        document_ids=session.get("document_ids", []),
        document_filename=session.get("document_filename"),
        preview="Empty conversation",
    )


@router.get("/sessions", response_model=list[ChatSession])
async def list_sessions():
    """List all active chat sessions."""
    sessions = list_all_sessions()
    return [
        ChatSession(
            session_id=s["session_id"],
            created_at=s["created_at"],
            message_count=s["message_count"],
            document_id=s.get("document_id"),
            document_ids=s.get("document_ids", []),
            document_filename=s.get("document_filename"),
            preview=s.get("preview", "Empty conversation"),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Fetch conversation history for a session."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = get_session_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(**m) for m in messages],
        document_id=session.get("document_id"),
        document_filename=session.get("document_filename"),
    )


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/{session_id}/message", response_model=ChatMessageResponse)
async def send_message(session_id: str, request: ChatMessageRequest):
    """
    Send a user message. Triggers hybrid search + LLM answer with citations.
    Non-streaming version — returns the full response at once.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    trace_id = str(uuid.uuid4())
    
    result = await process_message(
        session_id=session_id,
        user_message=request.message,
        trace_id=trace_id,
        metadata_filters=request.metadata_filters,
    )
    
    return ChatMessageResponse(
        session_id=session_id,
        message=ChatMessage(**result),
        trace_id=trace_id,
    )


@router.get("/{session_id}/stream")
async def stream_message(session_id: str, message: str = "", filters: str = ""):
    """
    Stream the LLM response token-by-token using Server-Sent Events (SSE).
    
    Query params:
    - message: the user's question
    - filters: JSON string of metadata filters (optional)
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    trace_id = str(uuid.uuid4())
    
    # Parse filters
    metadata_filters = {}
    if filters:
        try:
            metadata_filters = json.loads(filters)
        except json.JSONDecodeError:
            pass
    
    async def event_generator():
        try:
            # Send trace_id first
            yield f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id})}\n\n"
            
            async for event in process_message_stream(
                session_id=session_id,
                user_message=message,
                trace_id=trace_id,
                metadata_filters=metadata_filters,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            # Always send error + done events so the frontend never gets stuck
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
