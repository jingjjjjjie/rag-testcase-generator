"""
Task Management Router.

Provides endpoints for querying task status, retrieving results,
and managing background pipeline tasks.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from api.models import TaskStatusResponse, TaskResultResponse, TaskStatus, TaskListResponse
from api.task_manager import task_manager
import os

router = APIRouter()


@router.get("/", response_model=TaskListResponse)
async def list_tasks():
    """
    Get a list of all tasks.

    Returns:
        TaskListResponse with list of all tasks and their basic info
    """
    tasks = task_manager.list_all_tasks()

    # Format datetime objects to ISO strings
    for task in tasks:
        task["created_at"] = task["created_at"].isoformat()
        task["started_at"] = task["started_at"].isoformat() if task["started_at"] else None
        task["completed_at"] = task["completed_at"].isoformat() if task["completed_at"] else None

    return TaskListResponse(
        total=len(tasks),
        tasks=tasks
    )


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """
    Get current status and progress of a task.

    Args:
        task_id: UUID of the task

    Returns:
        Task status as plain dict (non-blocking)

    Raises:
        HTTPException: 404 if task not found
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Calculate progress from stages
    if task["stages"]:
        progress = sum(stage["progress"] for stage in task["stages"].values()) / 6.0
    else:
        progress = 0.0

    # Convert datetime objects to ISO format
    stages_formatted = {
        name: {
            **stage,
            "start_time": stage["start_time"].isoformat() if stage["start_time"] else None,
            "end_time": stage["end_time"].isoformat() if stage["end_time"] else None
        }
        for name, stage in task["stages"].items()
    }

    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "current_stage": task["current_stage"],
        "progress": progress,
        "stages": stages_formatted,
        "created_at": task["created_at"].isoformat(),
        "started_at": task["started_at"].isoformat() if task["started_at"] else None,
        "completed_at": task["completed_at"].isoformat() if task["completed_at"] else None,
        "total_tokens": task["total_prompt_tokens"] + task["total_completion_tokens"],
        "error": task["error"]
    }


@router.get("/{task_id}/result", response_model=TaskResultResponse)
async def get_task_result(task_id: str):
    """
    Get final results of a completed task.

    Args:
        task_id: UUID of the task

    Returns:
        TaskResultResponse with complete pipeline results

    Raises:
        HTTPException: 404 if task not found, 400 if task not completed
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed yet. Current status: {task.status}"
        )

    result = task_manager.get_result(task_id)

    if not result:
        raise HTTPException(status_code=404, detail="Task result not found")

    return TaskResultResponse(
        task_id=task_id,
        status=task.status,
        total_chunks=result["total_chunks"],
        total_facts=result["total_facts"],
        total_questions_generated=result["total_questions_generated"],
        total_valid_questions_extracted=result["total_valid_questions_extracted"],
        total_prompt_tokens=result["total_prompt_tokens"],
        total_completion_tokens=result["total_completion_tokens"],
        results=result["data"],
        extracted_questions=result["extracted_questions"]
    )


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    Request cancellation of a running task.

    The task will stop at the next stage boundary (between pipeline stages).
    Already completed stages will not be rolled back.

    Args:
        task_id: UUID of the task to cancel

    Returns:
        Success message

    Raises:
        HTTPException: 404 if task not found, 400 if task already completed
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task with status: {task['status']}"
        )

    task_manager.request_cancel(task_id)

    return {
        "message": f"Cancellation requested for task {task_id}. Task will stop at next stage boundary.",
        "task_id": task_id
    }


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task and its results from memory.

    Useful for cleaning up completed tasks and freeing memory.

    Args:
        task_id: UUID of the task to delete

    Returns:
        Success message

    Raises:
        HTTPException: 404 if task not found
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Remove from task manager
    with task_manager.lock:
        if task_id in task_manager.tasks:
            del task_manager.tasks[task_id]
        if task_id in task_manager.results:
            del task_manager.results[task_id]
        if task_id in task_manager.cancel_flags:
            del task_manager.cancel_flags[task_id]

    return {"message": f"Task {task_id} deleted successfully"}


@router.get("/{task_id}/download/full")
async def download_full_output(task_id: str):
    """
    Download the full pipeline output JSON file.

    Args:
        task_id: UUID of the task

    Returns:
        FileResponse with the full_output.json file

    Raises:
        HTTPException: 404 if task not found or file doesn't exist
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed yet. Current status: {task['status']}"
        )

    result = task_manager.get_result(task_id)

    if not result or "full_output_path" not in result:
        raise HTTPException(status_code=404, detail="Full output file path not found")

    file_path = result["full_output_path"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    filename = os.path.basename(file_path)

    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=filename
    )


@router.get("/{task_id}/download/questions")
async def download_extracted_questions(task_id: str):
    """
    Download the extracted questions JSON file.

    Args:
        task_id: UUID of the task

    Returns:
        FileResponse with the extracted_questions.json file

    Raises:
        HTTPException: 404 if task not found or file doesn't exist
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed yet. Current status: {task['status']}"
        )

    result = task_manager.get_result(task_id)

    if not result or "extracted_output_path" not in result:
        raise HTTPException(status_code=404, detail="Extracted questions file path not found")

    file_path = result["extracted_output_path"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    filename = os.path.basename(file_path)

    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=filename
    )
