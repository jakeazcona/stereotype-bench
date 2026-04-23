"""Task abstraction. A task is a generator of TaskPrompt instances."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from ..stimuli.base import Stimulus
from ..types import Message


@dataclass
class TaskPrompt:
    prompt_id: str
    stimulus: Stimulus
    messages: list[Message]
    metadata: dict = field(default_factory=dict)


class Task(Protocol):
    task_id: str

    def prompts(self) -> Iterable[TaskPrompt]: ...

    def clean_output(self, text: str) -> str:
        """Task-specific post-processing of raw model output before scoring.

        Default behavior for tasks that don't implement this: return text
        as-is. Implementations typically strip Markdown formatting or task-
        specific preambles so the scoring measure sees clean content.
        """
        ...
