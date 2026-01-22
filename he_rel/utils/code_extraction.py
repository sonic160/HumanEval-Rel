from __future__ import annotations

import re

def extract_fun_from_python_code(raw_completion: str, entry_point: str) -> str:
    """Extract the LAST top-level function named `fun_name` from `code`.

    Returns the function definition and body as a string, dedented to be
    a valid top-level function. If the function is not found an empty string
    is returned.

    We also gather any import statements and place them at the top of the output.
    """
    lines = raw_completion.splitlines()
    imports: list[str] = []
    result_lines: list[str] = []
    func_indent: int = 0

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)  # Store dedented imports
            i += 1
            continue
        if stripped.startswith("def "):
            name_part = stripped[len("def "):]
            indent_level = len(line) - len(stripped)
            if name_part.startswith(f"{entry_point}("):
                # Found the function - store lines and remember the indent
                func_indent = indent_level
                result_lines = [line]
                i += 1
                while i < n:
                    line = lines[i]
                    if line.strip() == "":
                        result_lines.append(line)
                        i += 1
                        continue
                    cur_indent = len(line) - len(line.lstrip())
                    if cur_indent <= indent_level:
                        break
                    result_lines.append(line)
                    i += 1
                continue
        i += 1

    if not result_lines:
        return ""

    # Dedent the function if it was indented 
    if func_indent > 0:
        dedented_lines = []
        for line in result_lines:
            if line.strip() == "":
                dedented_lines.append("")
            elif len(line) >= func_indent and line[:func_indent].strip() == "":
                dedented_lines.append(line[func_indent:])
            else:
                # Line has less indentation than expected, keep as is
                dedented_lines.append(line.lstrip())
        result_lines = dedented_lines

    # Build output: imports + blank line + function
    out = "\n".join(imports + [""] + result_lines) if imports else "\n".join(result_lines)

    return out



def extract_markdown_block(
        raw_completion: str,
        entry_point: str
    ) -> str:
        """Extract the last Python code block from a markdown string.

        A Python code block is defined as triple backticks followed by 'python'
        and ending with triple backticks.

        If no such block is found, returns an empty string.
        """
        pattern = r'```python\s*\n(.*?)\n```'
        matches: list[str] = re.findall(pattern, raw_completion, re.DOTALL)
        matches: filter[str] = filter(lambda code: f'def {entry_point}(' in code, matches)
        matches: list[str] = list(matches)
        if matches:
                return matches[-1]
        return ""