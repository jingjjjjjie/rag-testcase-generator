"""
Single-hop pipeline endpoints.
"""
from fastapi import APIRouter
from api.models import TaskSubmitResponse, TaskType
from api.task_manager import task_manager
from api.pipeline_runner import run_pipeline_with_tracking
import threading

router = APIRouter()


@router.post("/run", response_model=TaskSubmitResponse)
async def run_single_hop():
    """
    Submit a single-hop pipeline task to run in a separate thread.
    Returns immediately with task_id.

    The pipeline runs in its own thread, completely independent of FastAPI,
    so GET /tasks/{task_id} will always return immediately with current state.

    Returns:
        TaskSubmitResponse with task_id and status
    """
    # Create task with SINGLE_HOP type
    task_id = task_manager.create_task(task_type=TaskType.SINGLE_HOP)

    # Run in separate thread - this won't block FastAPI at all
    thread = threading.Thread(
        target=run_pipeline_with_tracking,
        args=(task_id,),
        daemon=True,
        name=f"Pipeline-{task_id[:8]}"
    )
    thread.start()

    return TaskSubmitResponse(
        task_id=task_id,
        status="submitted",
        message=f"Task {task_id} submitted successfully. Use /api/v1/tasks/{task_id} to check status."
    )
