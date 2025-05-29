from typing import Optional
import functools
import re

def extract_last_function(llm_completion: str) -> Optional[str]:
    """
    Extracts the last non-indented function definition from the text.

    Args:
        text (str): The text containing Python code.

    Returns:
        str: The extracted function code, or None if no function is found.
    """
    lines = llm_completion.split("\n")

    # Find the last non-indented function definition
    last_def_index = -1
    for i, line in enumerate(lines):
        if line.startswith("def "):
            last_def_index = i

    if last_def_index == -1:
        return None  # No function found

    # Determine the end of the function by checking the indentation
    # A function ends when a line with 0 indentation is encountered
    # that is not empty or a comment
    end_index = len(lines)
    for i in range(last_def_index + 1, len(lines)):
        line = lines[i].rstrip()
        if (
            line
            and not line.startswith(" ")
            and not line.startswith("\t")
            and not line.startswith("#")
        ):
            end_index = i
            break
    # a priori, models do not provide text after the function definition so end_index will probably be the end of the text, we keep this just in case
    return "\n".join(lines[last_def_index:end_index])



def extract_pattern(text :str, start:str, end:str) -> Optional[str]:
    pattern = re.escape(start) + r"\s*(.*?)" + re.escape(end)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None


extract_answer_block = functools.partial(extract_pattern, start="<answer>", end="</answer>")
extract_markdown_block = functools.partial(extract_pattern, start="```python", end="```")
