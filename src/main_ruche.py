import os

import gc
import torch
from transformers import BitsAndBytesConfig

# Custom imports.
from helpers.reproducibility import set_random_seeds
import core.model as model
import core.prompter as prompter
import benchmarking.benchmarker as benchmarker

from utils.taskmanagement import SlurmTask

# Reproducibility.
# set_random_seeds()
# login ('YOUR_TOKEN_HERE') # Only for models that you need access for (e.g LLama)


cache_dir = "/gpfs/workdir/baudoinso/.cache/huggingface"

def benchmark(model_name : str) -> None:
    print(f"Now benchmarking the following model : {model_name}")
    std_name = model_name.replace('/',"")
   
    # We try to see if the GPU can fit the whole model or if it should be quantized
    # Currently loading a model that is to big will end up in slurm killing the process
    # We have to manually declare models which have to be quantized

    
    # # We call the prompter over the model, save the completions in a file
    completion_file = f'./experiment_completions/completions_{model_name.replace("/", "")}.json'
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
        )

        quantization_config = None
        if model_name in to_quantize:
            quantization_config = BitsAndBytesConfig(
             load_in_4bit=True,
             bnb_4bit_quant_type="nf4",
             bnb_4bit_compute_dtype=torch.float16,
            )
        

        # # We initialize the model
        current_model = model.HuggingFace(
            std_name,
            model_name,
            quantization_config=quantization_config,
            cache_dir=cache_dir
            )
        current_model.model_init()    
        prompteur = prompter.Prompter(
            current_model,
            './data_set/benchmark.json',
            savepath=completion_file,
            batch=False, # True,
            n=10,
            # batch_size=256,
            stream=True,
            )
        # We clear memory 
        del prompteur
        del current_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else ()
        gc.collect()
    
    # With the completion's file, it run the functions and compute the score
    benchmarkeur = benchmarker.Benchmarker(
        './data_set/benchmark.json',
        completion_file=completion_file,
        timeout_warnings=True,
        )
    print(f"End of {model_name}'s benchmark") 
    
    #We clear memory
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
        "croissantllm/CroissantLLMBase",
        #"Qwen/Qwen2.5-Coder-7B-Instruct",
        #"mistralai/Ministral-8B-Instruct-2410",
        #"meta-llama/Llama-Guard-3-8B",  # (1)
        #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        #"princeton-nlp/gemma-2-9b-it-SimPO",
        #"google/gemma-2-9b-it",
        #"microsoft/Orca-2-13b" "google/gemma-2-27b-it",
        #"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        #"Qwen/Qwen2.5-Coder-32B-Instruct",  # (2)
        # "nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
        # "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        # "mistralai/Mistral-Large-Instruct-2407",
        # "openai-community/roberta-large-openai-detector"
        # "deepseek-ai/DeepSeek-V2.5",
    ]

    # We iterate over all the models
    for model_name in models_list:
        SlurmTask(task=lambda : benchmark(model_name), 
                  check_execution=lambda : os.path.isfile(f"./generated_data/experiment_results/results_{model_name.replace("/", "")}.json"),
                  name=f'benchmark{model_name}',
                  ).execute()
