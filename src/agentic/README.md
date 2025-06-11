# Agentic Code Generation and Knowledge Retrieval Framework

This project implements an Agentic Model combining reasoning, coding, and knowledge retrieval agents powered by large language models (LLMs). It is designed to generate code, answer questions, and summarize information through modular components.

## Example Usage

```py
from agentic import AgenticModel, HuggingFace, LitellmModel, EmbeddingModel

main_llm_name = "princeton-nlp/gemma-2-9b-it-SimPO"
embedding_name="Alibaba-NLP/gte-base-en-v1.5"

main_llm=HuggingFace(model_name=main_llm_name ,model_path=main_llm_name, cache_dir=cache_dir)
# main_llm=LitellmModel(model_name=main_llm_name, api_token=...)

challenge = """
def expected_number_of_failures(reliability: callable, t: float) -> float:
    ''' Given a reliability function (reliability) of a lifetime distribution, calculate the expected number of failures before time t, if a failed item is replaced imediately with the same one, and the replacement time can be negligible.
    Example:
    >>> expected_number_of_failures(lambda t: np.exp(-.001*t), 4320)
    4.32
    '''
"""
model = AgenticModel(
    "agentic",
    resoning_model=main_llm,
    coding_medel=main_llm,
    knowledge_model=main_llm,
    summarizer_model=main_llm,
    selector_model=EmbeddingModel(embedding_name),
    use_rag=True,
    selection="emb"
)

# selection="emb": Enables embedding-based document selection when using RAG.
# The alternative, "freq", uses keyword frequency (TF-IDF) and does not rely on the selector model.
# For more details, refer to ./agents/rag.py


completion, reasoning = model.generate(challenge)
print(f'{completion = }')
print(f'{reasoning = }')
```

## Components Overview
`AgenticModel`: Coordinates reasoning steps.

`CodingAgent`: Generates Python functions from signatures and instructions. Extracts and returns clean Python code from model outputs.

`GenerationModel`: Abstract base class for LLM wrappers. It has two concrete implementations `HuggingFace` for a localy hosted huggingface model, and `LitellmModel` to use an LLM through an API (GPT-4o-mini, ...)

`EmbeddingModel`: Generates dense vector embeddings for texts. Supports retrieval-augmented generation with RAG.

## 🧪 Benchmarking `gpt-4o-mini` and Princeton Models

This guide explains how to benchmark agentic models with and without RAG, using Hugging Face and OpenAI models on the dataset.

---

### ✅ Environment Setup

Ensure you're on a **GPU A100 node**, then run the following commands:

```bash
module purge
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/11.8.0/gcc-11.2.0
source activate benchmark

cd HumanEval-Rel1

```
### Benchmarking Modes (n=1, temperature=0.3)
🧠 Agentic Only (No RAG)

```bash
python src/benchmark_agentic.py "princeton-nlp/gemma-2-9b-it-SimPO" -n 1 -t 0.3

```
🧠 Agentic + RAG (Embedding-based)

```bash
python src/benchmark_agentic.py "princeton-nlp/gemma-2-9b-it-SimPO" -n 1 -t 0.3 -f True
```

🤖 GPT-4o-mini (Agentic + RAG)

```bash
python src/benchmark_gpt.py  -n 1 -t 0.3 -f True
```

N.B : Before running benchmark_gpt.py, make sure to:

✏️ Replace the first line of src/benchmark_gpt.py with your actual OpenAI API token.

