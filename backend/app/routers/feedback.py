"""Feedback and observability API endpoints (Phase 6)."""

import logging
from fastapi import APIRouter, HTTPException

from app.services.tracing import log_feedback, get_trace
from app.models.schemas import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit thumbs-up/down on an answer. Logs full prompt + context on downvote."""
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'")

    log_feedback(
        trace_id=request.trace_id,
        session_id=request.session_id,
        rating=request.rating,
        comment=request.comment or "",
    )

    return FeedbackResponse(
        status="ok",
        message=f"Feedback recorded: {request.rating}",
    )


@router.get("/traces/{trace_id}")
async def get_pipeline_trace(trace_id: str):
    """Fetch the full logged pipeline trace for a given request."""
    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
