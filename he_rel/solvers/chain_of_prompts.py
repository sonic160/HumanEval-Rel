"""Linear 4-step prompt chain solver for Inspect AI."""
from inspect_ai.solver import chain, generate, user_message, solver, Solver

from itertools import chain as iterchain
from pathlib import Path
import yaml


"""Prompts for the 4-step linear prompt chain workflow.

The prompts are loaded from an adjacent YAML file (`chain_of_prompt.yaml`).
If PyYAML is not installed or the file is missing/corrupt, a built-in
fallback dictionary (the original prompts) is used.
"""

_PROMPT_FILE = Path(__file__).with_suffix('.yaml')
STEPS: list[str] = ["reformulate", "math_model", "formal_solution", "python_implementation"]

def _load_prompts() -> dict[str, str]:

    with _PROMPT_FILE.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        if all(key in data for key in STEPS):
            return data
        else:
            raise ValueError("YAML file is missing required prompt keys.")


PROMPTS = _load_prompts()

def prompt_step(key: str) -> tuple[Solver, Solver]:
    return (user_message(PROMPTS[key]), generate())


@solver
def chain_of_prompts() -> Solver:
    keys: list[str] = ["reformulate", "math_model", "formal_solution", "python_implementation"]
    return chain(*iterchain.from_iterable(map(prompt_step, keys)))