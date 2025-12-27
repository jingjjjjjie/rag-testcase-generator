"""
Health check endpoint.
"""
from fastapi import APIRouter
from api.models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
