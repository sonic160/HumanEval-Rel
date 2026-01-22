from typing import Callable

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,

)
from inspect_ai.solver import TaskState

from he_rel.scoring.sandbox import SandboxCodeRunner
from he_rel.utils.code_extraction import extract_markdown_block

TIMEOUT_SECONDS = 5
SANDBOX = SandboxCodeRunner(timeout_seconds=TIMEOUT_SECONDS)


# Adapted from https://github.com/UKGovernmentBEIS/inspect_evals
@scorer(metrics=[accuracy(), stderr()])
def verify(
    extractor: Callable[[str, str], str] = extract_markdown_block,
) -> Scorer:
    """
    Scorer for HumanEval tasks. Verifies the correctness of generated code
    by executing it against the provided test cases in a sandboxed environment.

    Returns:
        Scorer: The verification scorer function.
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Score a model's output by running the generated code and test cases.

        Args:
            state (TaskState): The current task state containing model output and metadata.
            target (Target): The target output (not used).

        Returns:
            Score: The result of the verification, including correctness and explanation.
        """
        tests, entry_point = state.target
        answer_code = extractor(state.output.completion, entry_point)
        passed, error = SANDBOX.run_tests(answer_code, tests, entry_point)

        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=answer_code,
            explanation="".join(
                ["The following verification code was executed:\n\n"]
                + ["```python\n\n"]
                + [answer_code]
                + ["\n```"]
                + (
                    [f"\nThe submission was incorrect\n\n{error}"]
                    if not passed
                    else [""]
                )
            ),
        )

    return score

def build_test_code(
        completion: str, tests: str, entry_point: str
    ) -> str:
        code = (
            "from typing import List, Callable\n"
            + "import numpy as np\n"
            + str(completion)
            + "\n"
            + tests
            + "\ntest = check("
            + entry_point
            + ")\n"
        )
        return code