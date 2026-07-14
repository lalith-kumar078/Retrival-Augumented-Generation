"""Agentic tool API endpoints (Phase 5)."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.tools import web_search, calculate, summarize_document, plan_and_execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class CalculateRequest(BaseModel):
    expression: str


@router.post("/web-search")
async def tool_web_search(request: WebSearchRequest):
    """Invoke the web search tool directly."""
    results = await web_search(request.query, request.max_results)
    return {"query": request.query, "results": results}


@router.post("/calculate")
async def tool_calculate(request: CalculateRequest):
    """Code interpreter / calculator tool."""
    result = await calculate(request.expression)
    if result["error"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
