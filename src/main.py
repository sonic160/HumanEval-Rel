import os

import gc
from huggingface_hub import login
import torch
from transformers import BitsAndBytesConfig

# Custom imports.
from helpers.reproducibility import set_random_seeds
import model
import prompter
import benchmarker

# Reproducibility.
set_random_seeds()

# login ('YOUR_TOKEN_HERE') # Only for models that you need access for (e.g LLama)

# If the file is executed, we run the following code
if __name__ == "__main__":
    # A list of the models that will be benchmarked
    models_list = [
        # "facebook/MobileLLM-125M",
        # "facebook/MobileLLM-1B",
        #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "croissantllm/CroissantLLMBase",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        #"mistralai/Ministral-8B-Instruct-2410",
        #"meta-llama/Llama-Guard-3-8B",
        #"princeton-nlp/gemma-2-9b-it-SimPO",
        #"google/gemma-2-9b-it",
        #"microsoft/Orca-2-13b""google/gemma-2-27b-it",
        #"nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
        #"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        #"mistralai/Mistral-Large-Instruct-2407",
        #"openai-community/roberta-large-openai-detector" "deepseek-ai/DeepSeek-V2.5",
    ]
    # determine cache dir (test if we are in ruche)
    cache_dir = None
    if  "/gpfs/" in str(os.getcwd()):
        cache_dir = "/gpfs/workdir/baudoinso/.cache/huggingface"
    # We iterate over all the models
    for model_name in models_list : 

        # check if benchmark is already calculated for model and skip if neccessary
        std_name = model_name.replace('/',"")
        if os.path.isfile(f"./experiment_results/results_{std_name}.json"):
            print(f"{model_name}'s benchmark has already been calculated")
            #continue

        print(f"Now benchmarking the following model : {model_name}")
        # We try to see if the GPU can fit the whole model or if it should be quantized
        try:
            quantization_config = None

            current_model = model.HuggingFace(
                std_name,
                model_name,
                quantization_config=quantization_config,
                cache_dir=cache_dir
                )
            current_model.model_init()
        except Exception as e:
            print(e)
            # We offload the previous model from the memory
            del current_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else ()
            gc.collect()

            # We choose a quantization_config to use
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
                )
            current_model.model_init()
        
        # We call the prompter over the model, save the completions in a file
        completion_file=f'./data_set/json/completions_{current_model.model_name}.json'
        
        prompteur = prompter.Prompter(
            current_model,
            './data_set/benchmark.json',
            savepath=completion_file,
            batch=False, # True,
            n=50
            # batch_size=256,
            )
       
        # With the completion's file, it run the functions and compute the score
        benchmarkeur = benchmarker.Benchmarker(
            './data_set/benchmark.json',
            completion_file=completion_file,
            timeout_warnings=False
            )
        print(f"End of {model_name}'s benchmark") 
        
        # We clear memory for the following model
        del prompteur
        del current_model
        del benchmarkeur  
        torch.cuda.empty_cache() if torch.cuda.is_available() else ()
        gc.collect()