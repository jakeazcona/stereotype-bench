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
