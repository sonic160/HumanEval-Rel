import argparse
import time

parser = argparse.ArgumentParser("benchmarks a specific model")

parser.add_argument('model', type=str, help='The model to be tested')
parser.add_argument('-n', type=int, help='The number of answers to generate for every question', default=50)
parser.add_argument('-t', type=float, help='Temperature', default=0.3)
parser.add_argument('--max_tokens', type=int, help='newly_generated_tokens', default=1000)
parser.add_argument('-s', type=str, help='selection_rag', default="emb")
parser.add_argument('-f', type=bool, help='True: agentic+RAG , False : agentic_only', default=False)

args = parser.parse_args()

if args.n < 1:
    print("invalid value for number of samples -n")
    exit(1)
if args.t <= 0:
    print("invalid value for temperature -t")
    exit(1)

if args.max_tokens <= 0:
    print("invalid value for max_tokens -max_tokens")
    exit(1)


import os

# Custom imports.
from helpers.reproducibility import set_random_seeds
from agentic import AgenticModel, HuggingFace as GenerationModel, EmbeddingModel

from core import prompter
from benchmarking import benchmarker

# Reproducibility.
set_random_seeds()

embedding_name="Alibaba-NLP/gte-base-en-v1.5"

print(f"Benchmarking {args.model} with N={args.n} T={args.t} Max_Tokens={args.max_tokens}")

main_model_name=args.model

# determine cache dir (test if we are in ruche)
cache_dir = None
if  "/gpfs/" in str(os.getcwd()):
    cache_dir = "/gpfs/workdir/elkhattou2/.cache/huggingface"

main_model=GenerationModel(model_name=main_model_name,model_path=main_model_name, cache_dir=cache_dir,temperature=args.t,max_tokens=args.max_tokens)

if args.f:
    model_name = f"agentic_with_RAG_{main_model_name}_{embedding_name}"
else:
    model_name = f"agentic_{main_model_name}_{embedding_name}"

# check if benchmark is already calculated for model and skip if neccessary
std_name = model_name.replace("/", "")
if os.path.isfile(f"./experiment_results/results_{std_name}.json"):
    print(f"{model_name}'s benchmark has already been calculated")
    # continue


current_model = AgenticModel(
    "agentic_with_RAG",
    resoning_model=main_model,
    coding_medel=main_model,
    knowledge_model=main_model,
    summarizer_model=main_model,
    selector_model=EmbeddingModel(embedding_name),
    use_rag=args.f,
    rag_data_base_path='./data_set/dataset_reliability/RAG.txt',
    selection=args.s
)

now = time.time_ns()
completion_file = f"./data_set/json/completions_{std_name}_T={args.t}_N={args.n}_{now}.json"
results_file = f"./data_set/results/results_{std_name}_T={args.t}_N={args.n}_{now}.json"

# We call the prompter over the model, save the completions in a file
prompteur = prompter.Prompter(
    current_model,
    "./data_set/benchmark.json",
    savepath=completion_file,
    batch=False,  # True,
    n=args.n,
    temperature=args.t,
    max_tokens=args.max_tokens,
    # batch_size=256,
)

# With the completion's file, it runs the functions and computes the score
benchmarkeur = benchmarker.Benchmarker(
    "./data_set/benchmark.json",
    completion_file=completion_file,
    timeout_warnings=False,
    save_result_filepath=results_file,
)

print("saved results to:")
print(completion_file)
print(results_file)
print(f"End of {model_name}'s benchmark")