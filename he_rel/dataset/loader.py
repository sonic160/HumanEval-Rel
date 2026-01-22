import json
from pathlib import Path
from typing import Callable

from inspect_ai.dataset import Sample


def load_he_rel_dataset(json_filepath: Path, format_prompt: Callable[[str], str]) -> list[Sample]:
    """Load a HE-REL dataset from a JSONL file into `Sample` objects."""

    samples: list[Sample] = []
    with json_filepath.open("r", encoding="utf-8") as handle:
        json_content = json.load(handle)

    for sample in json_content:
        function_header = sample["prompt"]

        prompt = format_prompt(function_header)

        samples.append(
            Sample(
                id=sample["task_id"],
                input=prompt,
                target=[sample["tests"], sample["entry_point"]],
                metadata=sample.get("metadata", {}),
            )
        )

    return samples
