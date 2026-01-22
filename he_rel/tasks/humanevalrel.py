from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import Task, task
from inspect_ai.scorer import Scorer
from inspect_ai.solver import generate

from he_rel.dataset.loader import load_he_rel_dataset
from he_rel.scoring.code_tester import verify
from he_rel.utils.code_extraction import  extract_markdown_block
from he_rel.solvers.chain_of_prompts import chain_of_prompts


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "questions.jsonl"
)


def format_prompt(function_header: str) -> str:
    first_line = "def " +  function_header.split("def ")[1].split(":\n")[0] + ":"
    return f"""Complete this python function \n {function_header} \n Provide your final answer in a markdown code block. Format your response as: ```python\n{first_line}\n ### Your code here ###\n```"""


@task
def humanevalrel() -> Task:
    """Run a HumanEval-Rel code generation task."""

    dataset = load_he_rel_dataset(_DATASET_PATH, format_prompt)
    scorer: Scorer = verify(extractor=extract_markdown_block)

    return Task(
        dataset=dataset,
        scorer=scorer,
    )
