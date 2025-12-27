"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel
from typing import List, Dict, Any


class HealthResponse(BaseModel):
    status: str


class SingleHopResponse(BaseModel):
    status: str
    total_chunks: int
    total_facts: int
    results: List[Dict[str, Any]]


class MultiHopResponse(BaseModel):
    status: str
    message: str
