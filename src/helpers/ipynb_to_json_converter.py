import json
import re
import os
import sys
import argparse
import traceback
from functools import reduce

def extract_function_name(prompt):
    """Extract the function name from the prompt."""
    match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", prompt)
    if match:
        return match.group(1)
    return None

def process_notebook(notebook : tuple[dict, str]) -> list[dict]:
    """Process the notebook data and extract problems."""
    filename, notebook_data = notebook
    name = os.path.splitext(filename)[0]
    print(f"Processing {filename}...")
    results = []
    cells = notebook_data['cells']

    current_task_id = "error no task id found"
    current_prompt = None

    for i, cell in enumerate(cells):
        # Look for markdown cells with problem statements
        if cell['cell_type'] == 'markdown':
            content = ''.join(cell['source'])

            # Check if this is a question/problem cell
            if '# question' in content or '# prompt' in content.lower():
                # Extract the prompt (function definition with docstring)
                for regexp in (
                    r"```py\n(.*?)```",
                    r"``` py\n(.*?)```",
                    r"```python\n(.*?)```",
                    r"``` python\n(.*?)```",
                ):
                    code_block_match = re.search(regexp, content, re.DOTALL)
                    if code_block_match:
                        current_prompt = code_block_match.group(1).strip()
                        break
                    #

                question_number = re.search(
                    r"Question\s*(\d+)", content
                )
                if question_number:
                    current_task_id = f"{name}[Q{question_number.group(1)}]"

        # Look for code cells with unit tests
        # adapt for lower case
        elif cell['cell_type'] == 'code' and current_prompt and '# unit test' in ''.join(cells[i-1]['source']).lower():
            test_code = ''.join(cell['source'])

            # If we have both a prompt and test code, create an entry
            if current_prompt and test_code:
                entry_point = extract_function_name(current_prompt)
                if entry_point:
                    results.append({
                        'prompt': current_prompt+"\n",
                        'task_id': str(current_task_id),
                        'entry_point': entry_point,
                        'test': test_code,
                        'metadata': {
                            'origin': os.path.splitext(filename)[0],
                        }
                    })
                    print(f"    ->Extracted: {current_task_id}")
                    current_task_id = "error no task id found"
                    current_prompt = None

    return results

def main():
    parser = argparse.ArgumentParser(description='Extracts benchmark from a directory of Jupyter notebooks to JSON format')
    parser.add_argument('directory', help='Input Jupyter notebook files location')
    args = parser.parse_args()
    
    try:
        # Check if the directory exists
        if not os.path.isdir(args.directory):
            raise FileNotFoundError(f"Directory '{args.directory}' does not exist.")
        
        # Read the notebooks
        notebook_data_list = []
        notebooks = list(filter(lambda fp: fp.endswith(".ipynb"), os.listdir(args.directory)))
        notebooks.sort()
        for notebook_filepath in notebooks:
            with open(os.path.join(args.directory, notebook_filepath), 'r', encoding='utf-8') as f:
                try:
                    notebook_data = json.load(f)
                    notebook_data_list.append((notebook_filepath, notebook_data))
                except Exception as e:
                    print(f"    ->Error decoding JSON from {notebook_filepath}. Skipping this file.")
        # Process each notebook
        results = list(reduce(lambda x, y: x + y, map(process_notebook, notebook_data_list)))
        
        # Determine output JSON filepath
        output_file = os.path.join(args.directory, 'benchmark.json')
        # if the output file already exists, add (n) to the filename
        if os.path.exists(output_file):
            n = 1
            while os.path.exists(output_file):
                output_file = os.path.join(args.directory, f'benchmark({n}).json')
                n += 1
        # Write the results to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f: 
            json.dump(results, f, indent=4)


        print(f"Successfully converted notebooks in {args.directory} to {output_file}")
        print(f"Extracted {len(results)} problems.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("Notebook conversion failed.") 
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
