import os
import requests
# Custom imports.
from core.openrouter_model import OpenRouter
import core.prompter as prompter
import benchmarking.benchmarker as benchmarker


def benchmark(model_name: str, n : int = 1, max_tokens : int = 1000) -> None:
    print(f"Now benchmarking the following model : {model_name}")

    # We try to see if the GPU can fit the whole model or if it should be quantized
    # Currently loading a model that is to big will end up in slurm killing the process
    # We have to manually declare models which have to be quantized

    # # We call the prompter over the model, save the completions in a file
    std_name = model_name.replace("/", "")
    if ":" in std_name:
        std_name = std_name.split(":")[0]
        
    completion_file = f'./generated_data/experiment_completions/completions_{std_name}.json'
    if os.path.isfile(completion_file):
        print(
            f"Completion file {completion_file} already exists. Skipping generation.")
    else:
        quantization_config = None

        # # We initialize the model
        current_model = OpenRouter(
            model_name,
            api_key="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"
        )
        current_model.model_init()
        
        #generate completions
        prompter.Prompter(
            current_model,
            './data_set/benchmark.json',
            savepath=completion_file,
            batch=False,  # True,
            n=n,
            stream=True,
            parallel=True,
            max_tokens= max_tokens,
        )

    # With the completion's file, it run the functions and compute the score
    benchmarker.Benchmarker(
        './data_set/benchmark.json',
        completion_file=completion_file,
        timeout_warnings=True,
        parallel=True,
        N_WORKERS=None,  # None means use all available cores
    )
    print(f"End of {std_name}'s benchmark")



# If the file is executed, we run the following code
if __name__ == "__main__":

    # A list of the models that will be benchmarked
    models_list = [
         "openai/gpt-4.1-mini",
        #  "mistralai/devstral-small",
        #  "google/gemini-2.5-flash-preview-05-20",
        #  "qwen/qwen3-235b-a22b",
        #   "qwen/qwq-32b",
        #  "deepseek/deepseek-chat-v3-0324",
    ]

    n = int(input("Enter the number of requests to benchmark (n): ").strip())
    max_tokens = 2000
    # Get pricing information from OpenRouter API    
    try:
        print("Fetching model pricing data from OpenRouter API...")
        response = requests.get("https://openrouter.ai/api/v1/models")
        models_data = response.json()["data"]
        
        # Create a mapping of model names to pricing
        model_pricing = {}
        for model in models_data:
            model_pricing[model["id"]] = {
                "prompt": float(model["pricing"]["prompt"]),
                "completion": float(model["pricing"]["completion"]),
            }
        
        # Calculate estimated cost for selected models
        estimated_cost = 0
        for model_name in models_list:
            if model_name in model_pricing:
                # Rough estimate: assume 1k prompt tokens + 1k completion tokens per request
                cost_per_request = (
                    model_pricing[model_name]["prompt"] + model_pricing[model_name]["completion"]) * max_tokens
                estimated_cost += cost_per_request * n * 40 #dataset size is 40
                print(f"  {model_name}: ${cost_per_request:.6f} per request")
        
        print(f"Estimated total cost: ${estimated_cost:.4f}, n= {n}")
        
    except Exception as e:
        print(f"Could not fetch pricing data: {e}")
        estimated_cost = "unknown"
        exit(1)
    
    print("Are you sure you want to continue? (y/n)")
    if input().strip().lower() != 'y':
        print("Benchmarking aborted.")
        exit(0)
    for model_name in models_list:
        std_name = model_name.replace("/", "")
        if ":" in std_name:
            std_name = std_name.split(":")[0]
    
        benchmark(model_name+":nitro", n= n, max_tokens=max_tokens)
        