import os
import gc
import torch
from transformers import BitsAndBytesConfig

# Custom imports.
#from helpers.reproducibility import set_random_seeds
import core.model as model
import core.prompter as prompter
import benchmarker

from utils.taskmanagement import SlurmTask

# Reproducibility.
#set_random_seeds()
# login ('YOUR_TOKEN_HERE') # Only for models that you need access for (e.g LLama)


cache_dir = "/gpfs/workdir/baudoinso/.cache/huggingface"


def benchmark(model_name: str, temperature, max_tokens :int, prompt_prefix="", prompt_suffix : str="", completion_file : str = "./test_completion.json") -> None:
    
    print(f"Now benchmarking the following model : {model_name}")
    std_name = model_name.replace("/", "")

    # We try to see if the GPU can fit the whole model or if it should be quantized
    # Currently loading a model that is to big will end up in slurm killing the process
    # We have to manually declare models which have to be quantized

    #  We call the prompter over the model, save the completions in a file

    quantization_config = None

    to_quantize = (
        "nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "mistralai/Mistral-Large-Instruct-2407",
        "openai-community/roberta-large-openai-detector",
        "deepseek-ai/DeepSeek-V2.5",
    )

    quantization_config = None
    if model_name in to_quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # We initialize the model
    current_model = model.HuggingFace(
        std_name,
        model_name,
        quantization_config=quantization_config,
        cache_dir=cache_dir,
      
    )
    current_model.model_init()
    prompteur = prompter.Prompter(
        current_model,
        "./data_set/benchmark.json",
        savepath=completion_file,
        batch=False,  # True,
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=40,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        stream=True,
        show_example=False,
        # batch_size=256,
    )

    # We clear memory
    del prompteur
    del current_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else ()
    gc.collect()

    # With the completion's file, it run the functions and compute the score

    #  eventually fix older extraction issues
    # import json
    # content = None
    # with open(completion_file, "r") as f:
    #     content = json.loads(f.read())
    #     for i, test in enumerate(content):
    #         content[i]["completion"] = model.Model.extract_code(test["whole_answer"])
    # with open(completion_file, "w") as f:
    #     f.write(json.dumps(content))

    benchmarkeur = benchmarker.Benchmarker(
        "./data_set/benchmark.json",
        completion_file=completion_file,
        save_result_filepath=f"./test_results/results_{std_name}.json",
        timeout_warnings=True,
        model_name="test"
    )
    print(f"End of {model_name}'s benchmark")

    # We clear memory
    del benchmarkeur
    gc.collect()


# If the file is executed, we run the following code
if __name__ == "__main__":
    # determine cache dir (test if we are in ruche)
    IN_RUCHE = "/gpfs/" in str(os.getcwd())
    if not IN_RUCHE:
        raise Exception("You are not in ruche...")

    # A list of the models that will be benchmarked
    models_list = [
         "Qwen/QwQ-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ]
    temp = 0.6
    max_tokens = 10 * 1000
    prompt_prefix = """
I'll provide you with a function specification. Please implement it following these guidelines:

1. Think step-by-step about the solution approach and necessary algorithms
2. You may import standard libraries (numpy, scipy, collections, itertools, etc.) 
3. Any import or auxiliary function should be defined within the function, this way it will be self-contained
4. For numerical functions, ensure high accuracy but exact values aren't required
5. Optimize for readability

After your thinking process, provide only the complete function implementation with no explanations after.

Function specification:
"""
    prompt_suffix = "\n<think>"
    # We iterate over all the models
    for model_name in models_list:
        completion_path = f"./test_completions/test_completion_{model_name.replace('/', '')}t={temp}mt={max_tokens}.json"
        SlurmTask(
            task=lambda: benchmark(model_name, 
                                    temperature=temp,
                                    max_tokens=max_tokens,
                                    prompt_prefix=prompt_prefix,
                                    prompt_suffix=prompt_suffix,
                                    completion_file=completion_path),
            check_execution=lambda: False,
            name=f"benchmark {model_name}",
        ).execute()
