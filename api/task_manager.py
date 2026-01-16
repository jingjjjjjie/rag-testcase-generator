"""
Task Manager for handling background pipeline tasks.

Provides in-memory storage for task metadata, status, and results.
Thread-safe operations for concurrent task management.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from threading import Lock
from copy import deepcopy
from api.models import TaskInfo, TaskStatus, StageInfo, TaskType


class TaskManager:
    """
    Thread-safe in-memory task manager.
    Stores task metadata, status, and results.

    Note: All task data is stored in memory and will be lost on server restart.
    """

    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.cancel_flags: Dict[str, bool] = {}  # Track cancellation requests
        self.lock = Lock()

    def create_task(self, task_type: TaskType = TaskType.SINGLE_HOP) -> str:
        """
        Create a new task and return task_id.

        Args:
            task_type: Type of the task (SINGLE_HOP or MULTI_HOP)

        Returns:
            str: UUID of the newly created task
        """
        task_id = str(uuid.uuid4())

        with self.lock:
            self.tasks[task_id] = TaskInfo(
                task_id=task_id,
                task_type=task_type,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                stages={}
            )

        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task info by ID as plain dict (fastest, non-blocking).

        Args:
            task_id: UUID of the task

        Returns:
            Plain dict with task data or None if not found
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None

            # Convert to plain dict in one shot - no Pydantic overhead
            return {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "status": task.status.value,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "current_stage": task.current_stage,
                "stages": {
                    name: {
                        "name": stage.name,
                        "status": stage.status.value,
                        "progress": stage.progress,
                        "items_processed": stage.items_processed,
                        "items_total": stage.items_total,
                        "tokens_used": stage.tokens_used,
                        "start_time": stage.start_time,
                        "end_time": stage.end_time,
                        "error": stage.error
                    }
                    for name, stage in task.stages.items()
                },
                "total_prompt_tokens": task.total_prompt_tokens,
                "total_completion_tokens": task.total_completion_tokens,
                "error": task.error
            }

    def update_task_status(self, task_id: str, status: TaskStatus):
        """
        Update overall task status.

        Args:
            task_id: UUID of the task
            status: New status to set
        """
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status
                if status == TaskStatus.RUNNING and not self.tasks[task_id].started_at:
                    self.tasks[task_id].started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    self.tasks[task_id].completed_at = datetime.now()

    def update_stage(self, task_id: str, stage_name: str, stage_info: StageInfo):
        """
        Update specific stage information.

        Args:
            task_id: UUID of the task
            stage_name: Name of the pipeline stage
            stage_info: StageInfo object with updated information
        """
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].stages[stage_name] = stage_info
                self.tasks[task_id].current_stage = stage_name

    def update_tokens(self, task_id: str, prompt_tokens: int, completion_tokens: int):
        """
        Update token counts for a task.

        Args:
            task_id: UUID of the task
            prompt_tokens: Number of prompt tokens to add
            completion_tokens: Number of completion tokens to add
        """
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].total_prompt_tokens += prompt_tokens
                self.tasks[task_id].total_completion_tokens += completion_tokens

    def set_error(self, task_id: str, error_message: str):
        """
        Set error message and mark task as failed.

        Args:
            task_id: UUID of the task
            error_message: Error description
        """
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].error = error_message
                self.tasks[task_id].status = TaskStatus.FAILED
                self.tasks[task_id].completed_at = datetime.now()

    def store_result(self, task_id: str, result: Dict[str, Any]):
        """
        Store final pipeline results.

        Args:
            task_id: UUID of the task
            result: Dictionary containing pipeline results
        """
        with self.lock:
            self.results[task_id] = result

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored results for a task.

        Args:
            task_id: UUID of the task

        Returns:
            Results dictionary or None if not found
        """
        with self.lock:
            return self.results.get(task_id)

    def calculate_progress(self, task_id: str) -> float:
        """
        Calculate overall progress (0.0 to 1.0) based on stage completion.

        Args:
            task_id: UUID of the task

        Returns:
            float: Progress from 0.0 to 1.0
        """
        with self.lock:
            if task_id not in self.tasks:
                return 0.0

            task = self.tasks[task_id]
            if not task.stages:
                return 0.0

            # Single-hop has 6 stages, multi-hop has 8 stages
            num_stages = 8 if task.task_type == TaskType.MULTI_HOP else 6
            total_progress = sum(stage.progress for stage in task.stages.values())
            return total_progress / num_stages

    def request_cancel(self, task_id: str):
        """
        Request cancellation of a running task.

        Args:
            task_id: UUID of the task to cancel
        """
        with self.lock:
            if task_id in self.tasks:
                self.cancel_flags[task_id] = True

    def is_cancelled(self, task_id: str) -> bool:
        """
        Check if task has been cancelled.

        Args:
            task_id: UUID of the task

        Returns:
            bool: True if cancellation requested
        """
        with self.lock:
            return self.cancel_flags.get(task_id, False)

    def list_all_tasks(self) -> list:
        """
        Get a list of all tasks with their basic info.

        Returns:
            list: List of task dictionaries with basic information
        """
        with self.lock:
            tasks_list = []
            for task_id, task in self.tasks.items():
                # Calculate progress - single-hop has 6 stages, multi-hop has 8
                if task.stages:
                    num_stages = 8 if task.task_type == TaskType.MULTI_HOP else 6
                    progress = sum(stage.progress for stage in task.stages.values()) / num_stages
                else:
                    progress = 0.0

                tasks_list.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "status": task.status.value,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "current_stage": task.current_stage,
                    "progress": progress,
                    "error": task.error
                })

            # Sort by created_at, newest first
            tasks_list.sort(key=lambda x: x["created_at"], reverse=True)
            return tasks_list


# Global task manager instance
task_manager = TaskManager()
