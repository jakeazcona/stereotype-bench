from typing import Any

from .base import Task, TaskPrompt
from .first_impression import FirstImpressionTask

TASKS: dict[str, type] = {
    FirstImpressionTask.task_id: FirstImpressionTask,
}


def get_task(task_id: str, **task_params: Any) -> Task:
    """Instantiate a task by id, forwarding any extra kwargs to the constructor.

    The runner injects `measure=...` and `repetitions=...` via task_params; any
    additional task-specific options (variants, personas, primed_traits_n, ...)
    flow through as well.
    """
    if task_id not in TASKS:
        raise KeyError(f"Unknown task: {task_id!r}. Available: {list(TASKS)}")
    return TASKS[task_id](**task_params)


__all__ = ["Task", "TaskPrompt", "FirstImpressionTask", "TASKS", "get_task"]
