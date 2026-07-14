from fastapi import APIRouter
from app.services.usage import get_usage_stats
from app.config import settings

router = APIRouter(prefix="/usage", tags=["usage"])

@router.get("/stats")
async def get_stats():
    stats = get_usage_stats()
    
    # Try to determine a limit
    # The user asked to use GROQ_DAILY_TOKEN_LIMIT if possible.
    limit = getattr(settings, "groq_daily_token_limit", 500000) # Default fallback
    
    stats["daily_limit"] = limit
    return stats
