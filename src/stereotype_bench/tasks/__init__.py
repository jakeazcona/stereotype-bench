from .base import Task, TaskPrompt
from .first_impression import FirstImpressionTask

TASKS: dict[str, type] = {
    FirstImpressionTask.task_id: FirstImpressionTask,
}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task: {task_id!r}. Available: {list(TASKS)}")
    return TASKS[task_id]()


__all__ = ["Task", "TaskPrompt", "FirstImpressionTask", "TASKS", "get_task"]
