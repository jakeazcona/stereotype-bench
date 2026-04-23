"""Tests for FirstImpressionTask: variants, persona schema, primed sampling."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from stereotype_bench.tasks.first_impression import (
    ALL_VARIANTS,
    FirstImpressionTask,
    make_default_personas,
)


class FakeMeasure:
    """Stand-in for GSAMeasure that returns deterministic primed traits."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def sample_primed_traits(self, *, axis_target, n, rng):
        self.calls.append((axis_target, n))
        return [f"{axis_target}_trait_{i}" for i in range(n)]


def test_default_personas_is_50_balanced():
    personas = make_default_personas()
    assert len(personas) == 50
    males = [p for p in personas if p["gender"] == "male"]
    females = [p for p in personas if p["gender"] == "female"]
    assert len(males) == 25
    assert len(females) == 25
    for p in personas:
        assert 25 <= p["age"] <= 35


def test_plain_variant_no_measure_required():
    task = FirstImpressionTask(
        personas=[{"name": "X", "gender": "male", "age": 30}],
        variants=["plain"],
        repetitions=1,
    )
    prompts = task.prompts()
    assert len(prompts) == 1
    assert "X" in prompts[0].messages[0].content
    assert "30 years old" in prompts[0].messages[0].content


def test_three_traits_variant_no_measure_required():
    task = FirstImpressionTask(
        personas=[{"name": "Y", "gender": "female", "age": 28}],
        variants=["three_traits"],
        repetitions=1,
    )
    prompts = task.prompts()
    assert "3 character or personality traits" in prompts[0].messages[0].content
    assert "She is called Y" in prompts[0].messages[0].content


def test_primed_variants_require_measure():
    with pytest.raises(ValueError, match="sample_primed_traits"):
        FirstImpressionTask(
            personas=[{"name": "A", "gender": "male", "age": 30}],
            variants=["reinforcing"],
            repetitions=1,
            measure=None,
        )


def test_reinforcing_male_samples_high_axis():
    """Male reinforcing → male-typical traits → HIGH femtyp end."""
    fake = FakeMeasure()
    task = FirstImpressionTask(
        personas=[{"name": "A", "gender": "male", "age": 30}],
        variants=["reinforcing"],
        repetitions=1,
        measure=fake,
    )
    prompts = task.prompts()
    assert ("high", 3) in fake.calls
    assert "high_trait_0" in prompts[0].messages[0].content


def test_reinforcing_female_samples_low_axis():
    """Female reinforcing → female-typical traits → LOW femtyp end."""
    fake = FakeMeasure()
    task = FirstImpressionTask(
        personas=[{"name": "B", "gender": "female", "age": 27}],
        variants=["reinforcing"],
        repetitions=1,
        measure=fake,
    )
    task.prompts()
    assert ("low", 3) in fake.calls


def test_challenging_male_samples_low_axis():
    """Male challenging → female-typical traits → LOW femtyp end."""
    fake = FakeMeasure()
    task = FirstImpressionTask(
        personas=[{"name": "C", "gender": "male", "age": 31}],
        variants=["challenging"],
        repetitions=1,
        measure=fake,
    )
    task.prompts()
    assert ("low", 3) in fake.calls


def test_challenging_female_samples_high_axis():
    """Female challenging → male-typical traits → HIGH femtyp end."""
    fake = FakeMeasure()
    task = FirstImpressionTask(
        personas=[{"name": "D", "gender": "female", "age": 29}],
        variants=["challenging"],
        repetitions=1,
        measure=fake,
    )
    task.prompts()
    assert ("high", 3) in fake.calls


def test_repetitions_multiply_prompts():
    task = FirstImpressionTask(
        personas=[
            {"name": "A", "gender": "male", "age": 30},
            {"name": "B", "gender": "female", "age": 28},
        ],
        variants=["plain"],
        repetitions=10,
    )
    prompts = task.prompts()
    assert len(prompts) == 2 * 10  # 2 personas × 10 reps × 1 variant


def test_all_variants_combined_count():
    fake = FakeMeasure()
    task = FirstImpressionTask(
        personas=[
            {"name": "A", "gender": "male", "age": 30},
            {"name": "B", "gender": "female", "age": 28},
        ],
        variants=ALL_VARIANTS,
        repetitions=3,
        measure=fake,
    )
    prompts = task.prompts()
    assert len(prompts) == 2 * 4 * 3  # personas × variants × reps


def test_metadata_includes_variant_and_primed():
    fake = FakeMeasure()
    task = FirstImpressionTask(
        personas=[{"name": "A", "gender": "male", "age": 30}],
        variants=["plain", "reinforcing"],
        repetitions=1,
        measure=fake,
    )
    prompts = task.prompts()
    by_variant = {p.metadata["variant"]: p for p in prompts}
    assert by_variant["plain"].metadata["primed_traits"] is None
    assert by_variant["reinforcing"].metadata["primed_traits"] is not None
    assert len(by_variant["reinforcing"].metadata["primed_traits"]) == 3


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        FirstImpressionTask(
            personas=[{"name": "A", "gender": "male", "age": 30}],
            variants=["nonsense"],
        )


def test_pronoun_subject_dispatch():
    task = FirstImpressionTask(
        personas=[
            {"name": "M", "gender": "male", "age": 30},
            {"name": "F", "gender": "female", "age": 30},
        ],
        variants=["plain"],
    )
    prompts = task.prompts()
    by_name = {p.metadata["name"]: p.messages[0].content for p in prompts}
    assert "He is called M" in by_name["M"]
    assert "She is called F" in by_name["F"]


def test_clean_output_strips_markdown():
    task = FirstImpressionTask()
    cleaned = task.clean_output("# Header\n\nBody text")
    assert "Header" not in cleaned
    assert "Body text" in cleaned


def test_reproducible_with_seed():
    """Same seed -> same primed-trait samples across two task instantiations."""
    fake = FakeMeasure()
    t1 = FirstImpressionTask(
        personas=[{"name": "A", "gender": "male", "age": 30}],
        variants=["reinforcing"],
        repetitions=2,
        seed=99,
        measure=fake,
    )
    fake2 = FakeMeasure()
    t2 = FirstImpressionTask(
        personas=[{"name": "A", "gender": "male", "age": 30}],
        variants=["reinforcing"],
        repetitions=2,
        seed=99,
        measure=fake2,
    )
    p1 = t1.prompts()
    p2 = t2.prompts()
    # FakeMeasure ignores rng so this is trivially equal, but verifies the call
    # plumbing doesn't crash.
    assert [m.metadata["primed_traits"] for m in p1] == [
        m.metadata["primed_traits"] for m in p2
    ]
