import model
import prompter
import benchmarker
from huggingface_hub import login
from transformers import BitsAndBytesConfig
import torch

login ('INSERT YOU HF TOKEN HERE')

if __name__ == "__main__":
    models_list = ["croissantllm/CroissantLLMBase",
                   "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "meta-llama/CodeLlama-70b-hf",
                    "google/gemma-7b"
                    
    ]
    
    for model in models_list : 
        
        if model == "mistralai/Mixtral-8x7B-Instruct-v0.1" or "meta-llama/CodeLlama-70b-hf":
                quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=torch.float16,
                )
        else :
            quantization_config = None
        
        print(f"Now benchmarking the following model : {model}")
        modele = model.HuggingFace(model.replace('/',""),model, quantization_config=quantization_config)
        modele.model_init('../cache')

        prompteur = prompter.Prompter(modele, '../data_set/json/big.json', savepath= f'../data_set/json/completions_{modele.model_name}.json', batch=True, batch_size=256)
        
        benchmarkeur = benchmarker.Benchmarker('../data_set/json/big.json',f'../data_set/json/completions_{modele.model_name}.json')
        print(f"End of {model}'s benchmark")        
