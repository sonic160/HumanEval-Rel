import os

import gc
import torch
from transformers import BitsAndBytesConfig

# Custom imports.
import core.model as model
import core.prompter as prompter
import benchmarking.benchmarker as benchmarker

from utils.taskmanagement import SlurmTask
from utils.device import setup_device

# Import RAG components
from rag.ragmodel import RAGModel
from rag.database import FAISSDatabase
from rag.document_loaders import RagTextLoader, ReliawikiLoader

cache_dir = "/gpfs/workdir/baudoinso/.cache/huggingface"


def benchmark(
    model_name: str,
    temperature,
    max_tokens: int,
    prompt_prefix="",
    prompt_suffix: str = "",
    n: int = 1,
    completion_file: str = "./test_completion.json",
) -> None:
    print(f"Now benchmarking the following model : {model_name}")
    std_name = model_name.replace("/", "")

    # We try to see if the GPU can fit the whole model or if it should be quantized
    # Currently loading a model that is to big will end up in slurm killing the process
    # We have to manually declare models which have to be quantized


    if os.path.isfile(completion_file):
        print(f"Completion file {completion_file} already exists. Skipping generation.")
    else:
        quantization_config = None

        to_quantize = (
            "nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
            "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            "mistralai/Mistral-Large-Instruct-2407",
            "openai-community/roberta-large-openai-detector"
            "deepseek-ai/DeepSeek-V2.5",
            "Qwen/QwQ-32B"
        )

        quantization_config = None
        if model_name in to_quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Initialiser la base de données RAG
        document_loaders = [ReliawikiLoader("/gpfs/users/baudoinso/HumanEval-Rel/data/reliawiki_data.csv")]
        genai_cheatsheet = RagTextLoader(
            "./ragdata/genaicheatsheet.txt",
            encoding="utf-8",
        )
        database = FAISSDatabase(
            f"db_genai_cheatsheet", 
            [genai_cheatsheet],
            persist_directory=f"/generated_data/rag_databases/{std_name}"
        )

        # Initialiser le modèle RAG
        current_model = RAGModel(
            std_name,
            database, 
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
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=40,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            stream=True,
            show_example=False,
            attempt_recovery=True
            # batch_size=256,
        )
        # We clear memory
        del prompteur
        del current_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else ()
        gc.collect()

    # With the completion's file, it run the functions and compute the score
    benchmarkeur = benchmarker.Benchmarker(
        "./data_set/benchmark.json",
        completion_file=completion_file,
        timeout_warnings=True,
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
        #"croissantllm/CroissantLLMBase",
        # "mistralai/Ministral-8B-Instruct-2410",
        #"meta-llama/Llama-Guard-3-8B",  # (1)
         "Qwen/Qwen2.5-Coder-7B-Instruct",
         #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
         "princeton-nlp/gemma-2-9b-it-SimPO",
         "google/gemma-2-9b-it",
        # "microsoft/Orca-2-13b" "google/gemma-2-27b-it",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
         "Qwen/Qwen2.5-Coder-32B-Instruct",  # (2)
         #"Qwen/QwQ-32B",
        # "nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
        # "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        # "mistralai/Mistral-Large-Instruct-2407",
        # "openai-community/roberta-large-openai-detector"
        # "deepseek-ai/DeepSeek-V2.5",
    ]

    temp = 0.6
    max_tokens = 10000
    n=1
    prompt_prefix = """
I'll provide you with a function specification. Please implement it following these guidelines:

1. Think step-by-step about the solution approach and necessary algorithms
2. You may import standard libraries (numpy, scipy, collections, itertools, etc.) 
3. Any import or auxiliary function should be defined within the function, this way it will be self-contained
4. For numerical functions, ensure high accuracy but exact values aren't required
5. Optimize for readability

After your thinking process, provide only the complete function implementation with no explanations after.
"""
    prompt_suffix = "\n"
    # We iterate over all the models
    for model_name in models_list:
        completion_path = f"./generated_data/test_completion_{model_name.replace('/', '')}t={temp}mt={max_tokens}[RAG-reliawiki].json"
        SlurmTask(
            task=lambda: benchmark(
                model_name,
                temperature=temp,
                max_tokens=max_tokens,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                n=n,
                completion_file=completion_path,
            ),
            check_execution=lambda: False,
            name=f"benchmark {model_name}",
        ).execute()
