"""
Multi-hop pipeline endpoints.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from api.models import TaskSubmitResponse, TaskType
from api.task_manager import task_manager
from api.pipeline_runner_multi_hop import run_multi_hop_pipeline_with_tracking
import threading
from pathlib import Path
from typing import Optional
from src import PROJECT_ROOT

router = APIRouter()


@router.post("/run", response_model=TaskSubmitResponse)
async def run_multi_hop(file: Optional[UploadFile] = File(None)):
    """
    Submit a multi-hop pipeline task to run in a separate thread.
    
    Args:
        file: Optional PDF file to upload and process. If not provided,
              uses the PDF path from environment variables.

    Returns:
        TaskSubmitResponse with task_id and status
    """
    pdf_path = None

    # If file is uploaded, save it first
    if file:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Save to src/files directory
        upload_dir = Path(PROJECT_ROOT) / "src" / "files"
        upload_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = upload_dir / file.filename

        # Save uploaded file
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

    # Create task with MULTI_HOP type
    task_id = task_manager.create_task(task_type=TaskType.MULTI_HOP)

    # Run in separate thread
    thread = threading.Thread(
        target=run_multi_hop_pipeline_with_tracking,
        args=(task_id, str(pdf_path) if pdf_path else None),
        daemon=True,
        name=f"MultiHopPipeline-{task_id[:8]}"
    )
    thread.start()

    message = f"Multi-hop task {task_id} submitted successfully. Use /tasks/{task_id} to check status."
    if file:
        message = f"PDF '{file.filename}' uploaded and multi-hop task {task_id} submitted. Use /tasks/{task_id} to check status."

    return TaskSubmitResponse(
        task_id=task_id,
        status="submitted",
        message=message
    )
