"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


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


# Task Management Models

class TaskStatus(str, Enum):
    """Status of a background task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Type of pipeline task"""
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class StageInfo(BaseModel):
    """Information about a pipeline stage"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    items_processed: int
    items_total: int
    tokens_used: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None


class TaskInfo(BaseModel):
    """Complete information about a task"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_stage: Optional[str] = None
    stages: Dict[str, StageInfo] = {}
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    error: Optional[str] = None


class TaskSubmitResponse(BaseModel):
    """Response when submitting a new task"""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Response for task status query"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    status: TaskStatus
    current_stage: Optional[str] = None
    progress: float  # Overall progress 0.0 to 1.0
    stages: Dict[str, StageInfo]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tokens: int
    error: Optional[str] = None


class TaskResultResponse(BaseModel):
    """Response for completed task results"""
    task_id: str
    status: TaskStatus
    total_chunks: int
    total_facts: int
    total_questions_generated: int
    total_valid_questions_extracted: int
    total_prompt_tokens: int
    total_completion_tokens: int
    results: List[Dict[str, Any]]
    extracted_questions: List[Dict[str, Any]]


class TaskListItem(BaseModel):
    """Single task item in the list"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_stage: Optional[str] = None
    progress: float
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response for listing all tasks"""
    total: int
    tasks: List[TaskListItem]
