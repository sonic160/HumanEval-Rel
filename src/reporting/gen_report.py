import os
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List


@dataclass
class ModelResults:
    name: str
    result_file_path: str
    size: float = None  # in billion parameters
    is_open_source: bool = True
    __data: dict = None

    @property
    def results(self) -> dict:
        if self.__data is None:
            self.__load_data()
        return self.__data

    def __load_data(self) -> None:
        with open(self.result_file_path) as f:
            self.__data = json.load(f)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k using the unbiased estimator formula.
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    :return: pass@k value
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def generate_pass_at_k_report(models: List[ModelResults], output_dir: str, n: int):
    """
    Generates a Markdown report comparing the pass@k metrics for the given models.
    :param models: List of Model instances.
    :param output_dir: Directory where the report and images will be saved.
    :param n: Total number of samples per task.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    markdown_lines = ["# Pass@k Comparison Report", ""]

    # Add overview
    markdown_lines.append("## Overview")
    markdown_lines.append(
        "This report compares the Pass@k metrics across different tasks for the provided models."
    )
    markdown_lines.append("---")

    # General Pass@k graphs
    general_pass_at_k_path = output_path / "general_pass_at_k.png"
    plt.figure(figsize=(10, 6))

    for model in models:
        pass_per_k = model.results.get("pass_per_k", {})
        k_values = list(map(int, pass_per_k.keys()))[:-2]
        pass_at_k_values = list(pass_per_k.values())[:-2]
        print(f"{model.name} Pass@50 values: {pass_at_k_values[-1]}")
        plt.plot(k_values, pass_at_k_values, marker='o', label=model.name)

    plt.title("General Pass@k Comparison")
    plt.xlabel("k")
    plt.ylabel("Pass@k")
    plt.legend()
    plt.grid(True)
    plt.savefig(general_pass_at_k_path)
    plt.close()

    markdown_lines.append("## General Pass@k Graphs")
    markdown_lines.append("Below is the general Pass@k graph comparing the models.")
    markdown_lines.append(f"![General Pass@k Comparison](./{general_pass_at_k_path.name})")
    markdown_lines.append("---")

    # Combined histograms for each k value
    markdown_lines.append("## Pass@k Histograms")
    k_values = list(map(int, models[0].results.get("pass_per_k", {}).keys()))[:8]  # Assuming all models have the same k values

    for k in k_values[:-2]:
        hist_path = output_path / f"pass_at_{k}_comparison_histogram.png"

        # Prepare data for the histogram
        task_ids = None
        model_task_pass = {}
        for model in models:
            task_results = model.results.get("test_results", {})
            if task_ids is None:
                task_ids = list(task_results.keys())
            task_pass_at_k = []
            for task, completions in task_results.items():
                c = sum(1 for completion in completions if completion[0])  # Correct completions
                task_pass_at_k.append(pass_at_k(n, c, k))
            model_task_pass[model.name] = task_pass_at_k

        # Generate the combined histogram
        bar_width = 0.2
        x_positions = np.arange(len(task_ids))
        plt.figure(figsize=(12, 6))

        for i, (model_name, task_pass_at_k) in enumerate(model_task_pass.items()):
            plt.bar(
                x_positions + i * bar_width,
                task_pass_at_k,
                width=bar_width,
                label=model_name,
                alpha=0.7
            )

        plt.title(f"Pass@{k} Histogram Comparison Across Models")
        plt.xlabel("Tasks")
        plt.ylabel(f"Pass@{k}")
        plt.xticks(x_positions + (bar_width * (len(models) - 1) / 2), task_ids, rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        markdown_lines.append(f"### Pass@{k} Histogram")
        markdown_lines.append(f"![Pass@{k} Histogram](./{hist_path.name})")
        markdown_lines.append("---")

    # Save Markdown report
    report_path = output_path / "PassAtK_Report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(markdown_lines))

    print(f"Report generated at: {report_path}")

def generate_tex_code_file(models : List[ModelResults], output_dir : str) -> None:
    file_head =  "".join(open("report/tex_head.tex").readlines())
    file_tail = "".join(open("report/tex_tail.tex").readlines())
    body = ""  

    for model in models:
        generated_code = model.results.get('test_results', {})
        if generated_code == {}:
            continue

        model_tex = f"\\chapter[{model.name}]"+"{\\huge "+ model.name+"}\n"
        for question, attempts in generated_code.items():
            one_pass = False
            one_fail = False
            model_tex += '\\section{Question ' + question + '}\n'
            for i, attempt in enumerate(attempts):
                success, code = attempt
                if success:
                    if one_pass:
                        continue
                    one_pass = True
                else:
                    if one_fail and not code:
                        continue
                    one_fail = True
                
                if success:
                    model_tex += "\\captionsetup[lstlisting]{format=listingpass,labelfont=white,textfont=white}\n"
                else:
                    model_tex += "\\captionsetup[lstlisting]{format=listingfail,labelfont=white,textfont=white}\n"
                model_tex += "\\begin{lstlisting}[caption=Attempt \\#"+ f" {i} ({"PASS" if success else "FAIL"})]\n"
                model_tex += code + "\n"
                model_tex += "\\end{lstlisting}\n"
        body += model_tex

    # Write the complete LaTeX file
    with open(os.path.join(output_dir, "code.tex"), "w") as f:
        f.write(file_head + "\n" + body + "\n" + file_tail)


# Example usage
if __name__ == "__main__":
    models_list = [
       ## "facebook/MobileLLM-125M",
        ###"facebook/MobileLLM-1B",
        #"croissantllm/CroissantLLMBase",
        #"Qwen/Qwen2.5-Coder-7B-Instruct",
        #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        #"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        #"Qwen/Qwen2.5-Coder-32B-Instruct",
        #"openAI/gpt4omini",
        #"mistralai/Ministral-8B-Instruct-2410",
        #"meta-llama/Llama-Guard-3-8B",
        #"princeton-nlp/gemma-2-9b-it-SimPO",
        #"google/gemma-2-9b-it",
        #"microsoft/Orca-2-13b",
        #"google/gemma-2-27b-it",
        #"nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
        #"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        #"mistralai/Mistral-Large-Instruct-2407",
        #"openai-community/roberta-large-openai-detector" "deepseek-ai/DeepSeek-V2.5",
        "completion_meta-llamaLlama-Guard-3-8Bt=0.6mt=1000[RAG]",
        "completion_QwenQwen2.5-Coder-7B-Instructt=0.6mt=1000[RAG]",
    ]
    models = [
        ModelResults(name=model, 
              result_file_path=f"./generated_data/experiment_results/results_{model}.json"
            ) 
        for model in models_list 
        if os.path.isfile(f"./generated_data/experiment_results/results_{model}.json")
    ]
    # comparison for maxtoken
    
    # Assuming n = 50 for the total number of samples
    generate_pass_at_k_report(models, output_dir="./report", n=50)
    generate_tex_code_file(models, output_dir="./report")
