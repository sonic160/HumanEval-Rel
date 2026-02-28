"""V1 — Self-Debugging solver: V0 chain-of-prompts + execution feedback loop."""

from __future__ import annotations

from itertools import chain as iterchain
from pathlib import Path

import yaml

from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain,
    generate,
    solver,
    user_message,
)

from he_rel.scoring.sandbox import SandboxCodeRunner
from he_rel.utils.code_extraction import extract_markdown_block

_PROMPT_FILE = Path(__file__).with_suffix(".yaml")

STEPS: list[str] = [
    "reformulate",
    "math_model",
    "formal_solution",
    "python_implementation",
]

TIMEOUT_SECONDS = 5
SANDBOX = SandboxCodeRunner(timeout_seconds=TIMEOUT_SECONDS)


def _load_prompts() -> dict[str, str]:
    with _PROMPT_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


PROMPTS = _load_prompts()


def _build_test_code(completion: str, tests: str, entry_point: str) -> str:
    return (
        "from typing import List, Callable\n"
        "import numpy as np\n"
        + completion
        + "\n"
        + tests
        + "\ntest = check("
        + entry_point
        + ")\n"
    )


def _extract_math_model(messages: list) -> str:
    """Extract the math model response (assistant reply after step 2)."""
    # The math_model is the 4th message (index 3): system, user1, asst1, user2, asst2
    # With chain_of_prompts pattern: pairs of (user_message, generate) for each step
    # Messages: [user_prompt, asst_0, user_reformulate, asst_1, user_math_model, asst_2, ...]
    # The math model response is assistant message at index 4 (after 2 user + 2 assistant + 1 user)
    for i, msg in enumerate(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if "STEP 2" in msg.content and msg.role == "user":
                # The next assistant message is the math model
                if i + 1 < len(messages) and messages[i + 1].role == "assistant":
                    content = messages[i + 1].content
                    return content if isinstance(content, str) else str(content)
    return "(math model not found)"


@solver
def self_debugging(max_retries: int = 3, rubber_duck: bool = False) -> Solver:
    """V1 solver: chain-of-prompts (V0) followed by execution + retry loop.

    Args:
        max_retries: Maximum number of debug iterations.
        rubber_duck: If True, insert a rubber-duck review step before each fix.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # --- Phase 1: Run V0 chain-of-prompts (4 steps) ---
        for step_key in STEPS:
            state.messages.append(ChatMessageUser(content=PROMPTS[step_key]))
            state = await generate(state)

        # --- Phase 2: Execute and debug loop ---
        tests, entry_point = state.target
        problem = state.input_text

        for attempt in range(max_retries):
            # Extract code from the latest assistant message
            code = extract_markdown_block(state.output.completion, entry_point)
            if not code:
                # No code found — ask the model to provide it
                state.messages.append(
                    ChatMessageUser(
                        content=(
                            "Your response did not contain a valid Python code block "
                            f"with function `{entry_point}`. Please provide the "
                            "implementation in a ```python code block."
                        )
                    )
                )
                state = await generate(state)
                continue

            # Run tests
            test_code = _build_test_code(code, tests, entry_point)
            passed, error = SANDBOX.run_tests(code, tests, entry_point)

            if passed:
                break  # Tests pass — we're done

            # Retrieve math model for context
            math_model = _extract_math_model(state.messages)

            if rubber_duck:
                # Rubber duck step: explain code line by line
                rd_prompt = PROMPTS["rubber_duck"].format(
                    code=code, math_model=math_model
                )
                state.messages.append(ChatMessageUser(content=rd_prompt))
                state = await generate(state)
            else:
                # Standard debug prompt
                debug_prompt = PROMPTS["debug"].format(
                    problem=problem,
                    math_model=math_model,
                    code=code,
                    error=error or "Test assertion failed",
                )
                state.messages.append(ChatMessageUser(content=debug_prompt))
                state = await generate(state)

        return state

    return solve


@solver
def self_debugging_rd(max_retries: int = 3) -> Solver:
    """V1 variant with rubber duck review enabled."""
    inner = self_debugging(max_retries=max_retries, rubber_duck=True)
    return inner
