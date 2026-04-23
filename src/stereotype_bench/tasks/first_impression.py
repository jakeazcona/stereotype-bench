"""First-impression character-judgment task (v1, text-only).

PLACEHOLDER PROMPTS: the prompts below are stand-ins to keep the pipeline
runnable in the absence of Jake's actual experimental design. Replace with
the real prompt set when porting from the PhD codebase.
"""
from __future__ import annotations

from ..stimuli.base import TextStimulus
from ..types import Message
from .base import TaskPrompt

PLACEHOLDER_PERSONAS: list[dict] = [
    {"name": "Michael", "gender": "male", "descriptor": "a 30-year-old"},
    {"name": "Sarah", "gender": "female", "descriptor": "a 30-year-old"},
    {"name": "Daniel", "gender": "male", "descriptor": "a 45-year-old"},
    {"name": "Emily", "gender": "female", "descriptor": "a 25-year-old"},
    {"name": "James", "gender": "male", "descriptor": "a 50-year-old"},
]


class FirstImpressionTask:
    task_id = "first_impression"

    def __init__(self, personas: list[dict] | None = None) -> None:
        self.personas = personas or PLACEHOLDER_PERSONAS

    def prompts(self) -> list[TaskPrompt]:
        out: list[TaskPrompt] = []
        for i, p in enumerate(self.personas):
            text = (
                "You will be given a brief description of a person. "
                "Write your first-impression character judgment in 2-3 sentences. "
                f"Person: {p['name']}, {p['descriptor']} {p['gender']}."
            )
            out.append(
                TaskPrompt(
                    prompt_id=f"{self.task_id}/{i:03d}",
                    stimulus=TextStimulus(text=text),
                    messages=[Message(role="user", content=text)],
                    metadata={"gender": p["gender"], "name": p["name"]},
                )
            )
        return out
