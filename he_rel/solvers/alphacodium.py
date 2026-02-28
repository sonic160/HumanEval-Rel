"""V2 — AlphaCodium-style solver: enriched pre-processing + iterative test-anchored debugging."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)

from he_rel.scoring.sandbox import SandboxCodeRunner
from he_rel.utils.code_extraction import extract_markdown_block

_PROMPT_FILE = Path(__file__).with_suffix(".yaml")

TIMEOUT_SECONDS = 5
SANDBOX = SandboxCodeRunner(timeout_seconds=TIMEOUT_SECONDS)


def _load_prompts() -> dict[str, str]:
    with _PROMPT_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROMPTS = _load_prompts()


def _extract_test_assertions(text: str) -> list[str]:
    """Extract individual assert statements from the model's generated tests."""
    tests = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            tests.append(stripped)
    return tests


def _extract_section_response(messages: list, step_keyword: str) -> str:
    """Extract the assistant response following a user message containing step_keyword."""
    for i, msg in enumerate(messages):
        if (
            hasattr(msg, "role")
            and msg.role == "user"
            and hasattr(msg, "content")
            and isinstance(msg.content, str)
            and step_keyword in msg.content
        ):
            if i + 1 < len(messages) and messages[i + 1].role == "assistant":
                content = messages[i + 1].content
                return content if isinstance(content, str) else str(content)
    return ""


@solver
def alphacodium(max_iterations: int = 4) -> Solver:
    """V2 AlphaCodium-style solver.

    Phase A — Pre-processing (natural language reasoning):
      1. Self-reflection: reformulate, identify pitfalls, edge cases
      2. Solution candidates: propose 2-3 approaches, rank by robustness
      3. Generate additional tests

    Phase B — Iterative code generation with test anchoring:
      1. Generate initial code using top approach
      2. Run against public tests (docstring examples), debug if needed
      3. Run against AI-generated tests, debug with test anchoring
      4. Iterate up to max_iterations total

    Args:
        max_iterations: Max total debug iterations across both public and AI test phases.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tests, entry_point = state.target
        problem = state.input_text

        # =====================================================================
        # Phase A: Pre-processing (3 LLM calls, no code execution)
        # =====================================================================

        # Step 1: Self-reflection
        state.messages.append(ChatMessageUser(content=PROMPTS["self_reflection"]))
        state = await generate(state)
        self_reflection = state.output.completion

        # Step 2: Solution candidates
        state.messages.append(ChatMessageUser(content=PROMPTS["solution_candidates"]))
        state = await generate(state)

        # Step 3: Generate additional test cases
        state.messages.append(ChatMessageUser(content=PROMPTS["generate_tests"]))
        state = await generate(state)
        ai_tests_raw = state.output.completion
        ai_test_assertions = _extract_test_assertions(ai_tests_raw)

        # =====================================================================
        # Phase B: Iterative code + execution
        # =====================================================================

        # Step 4: Generate initial code
        state.messages.append(
            ChatMessageUser(content=PROMPTS["python_implementation"])
        )
        state = await generate(state)

        iterations_used = 0
        last_passing_code = None  # anchor: last code that passed public tests

        # --- Sub-phase B1: Debug against public tests ---
        while iterations_used < max_iterations:
            code = extract_markdown_block(state.output.completion, entry_point)
            if not code:
                state.messages.append(
                    ChatMessageUser(
                        content=(
                            f"Your response did not contain a valid Python code block "
                            f"with function `{entry_point}`. Please provide it in a "
                            "```python code block."
                        )
                    )
                )
                state = await generate(state)
                iterations_used += 1
                continue

            passed, error = SANDBOX.run_tests(code, tests, entry_point)

            if passed:
                last_passing_code = code
                break  # Public tests pass

            # Debug prompt for public test failure
            debug_prompt = PROMPTS["debug_public"].format(
                problem=problem,
                self_reflection=self_reflection,
                code=code,
                error=error or "Test assertion failed",
            )
            state.messages.append(ChatMessageUser(content=debug_prompt))
            state = await generate(state)
            iterations_used += 1

        # If we never passed public tests, return as-is
        if last_passing_code is None:
            return state

        # --- Sub-phase B2: Debug against AI-generated tests (with anchoring) ---
        for ai_test in ai_test_assertions:
            if iterations_used >= max_iterations:
                break

            # Build test code: the AI-generated assertion wrapped in a check function
            ai_test_code = (
                "from typing import List, Callable\n"
                "import numpy as np\n"
                + last_passing_code
                + "\n"
                + f"try:\n    {ai_test}\n    test = 'pass'\n"
                + "except Exception as e:\n    test = f'fail: {e}'\n"
            )

            passed_ai, error_ai = SANDBOX.execute(ai_test_code)

            if passed_ai:
                continue  # This AI test passes, move on

            # AI test failed — ask model to fix
            debug_prompt = PROMPTS["debug_ai_tests"].format(
                problem=problem,
                self_reflection=self_reflection,
                code=last_passing_code,
                failed_test=ai_test,
                error=error_ai or "Test assertion failed",
            )
            state.messages.append(ChatMessageUser(content=debug_prompt))
            state = await generate(state)
            iterations_used += 1

            # Extract candidate fix
            candidate_code = extract_markdown_block(
                state.output.completion, entry_point
            )
            if not candidate_code:
                continue

            # Test anchor: verify candidate still passes public tests
            passed_public, _ = SANDBOX.run_tests(
                candidate_code, tests, entry_point
            )
            if passed_public:
                # Also check it passes the AI test
                anchor_test_code = (
                    "from typing import List, Callable\n"
                    "import numpy as np\n"
                    + candidate_code
                    + "\n"
                    + f"try:\n    {ai_test}\n    test = 'pass'\n"
                    + "except Exception as e:\n    test = f'fail: {e}'\n"
                )
                passed_both, _ = SANDBOX.execute(anchor_test_code)
                if passed_both:
                    last_passing_code = candidate_code
                # else: candidate passes public but not AI test — keep old code
            # else: candidate breaks public tests — reject, keep anchor

        # Ensure the final output contains the best code
        if last_passing_code:
            final_msg = f"```python\n{last_passing_code}\n```"
            state.messages.append(ChatMessageAssistant(content=final_msg))
            state.output.completion = final_msg

        return state

    return solve
