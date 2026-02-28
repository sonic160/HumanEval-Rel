# Solvers — Prompting strategies for HumanEval-Rel

This folder contains the different solving strategies (solvers) for the HumanEval-Rel benchmark.
Each solver is registered as an Inspect AI solver usable via `--solver <name>`.

---

## Overall architecture

```
solvers/
├── chain_of_prompts.py      # V0 — Linear pipeline (baseline)
├── chain_of_prompts.yaml
├── self_debugging.py         # V1 — V0 + execution/correction loop
├── self_debugging.yaml
├── alphacodium.py            # V2 — AlphaCodium-style flow (preprocessing + iteration)
├── alphacodium.yaml
└── README.md                 # This file (French)
```

---

## V0 — Chain-of-Prompts linear (baseline)

**Solver:** `chain_of_prompts`
**Files:** `chain_of_prompts.py`, `chain_of_prompts.yaml`

Sequential pipeline with 4 LLM calls, no code execution:

1. **Reformulate** — The LLM reformulates the problem, clarifies constraints, inputs/outputs
2. **Math Model** — The LLM identifies variables, formulas, relevant algorithms
3. **Formal Solution** — The LLM lays out the mathematical solution step by step
4. **Python Implementation** — The LLM generates the final code

No feedback, no execution. A single linear pass. If the code is wrong, that's the end.

**LLM calls:** 4 (fixed)

```bash
uv run inspect eval ./he_rel/_registry.py@humanevalrel \
    --model openrouter/openai/gpt-4.1-mini:nitro \
    --solver chain_of_prompts
```

---

## V1 — Self-Debugging (V0 + execution + retry loop)

**Solver:** `self_debugging`
**Variant:** `self_debugging_rd` (with rubber duck review)
**Files:** `self_debugging.py`, `self_debugging.yaml`

We keep the V0 pipeline as the first pass, then add a correction loop:

1. **Steps 1→4 of V0** — Generation of the first code candidate
2. **Execution** — The code is run against the docstring tests via `SandboxCodeRunner`. We capture: success/failure, error message, obtained vs expected result
3. **Diagnostic** — If failure, a debug prompt is constructed containing:
   - The original problem (docstring)
   - The mathematical model produced in step 2
   - The generated code
   - The execution result (full traceback or "expected X but got Y")
4. **Retry** — The corrected code is re-run. Loop until success or max N=3 iterations

Rubber duck variant (`self_debugging_rd`): before retrying, the LLM explains its code line by line and checks coherence with the mathematical model. Targets "wrong answer" errors (dominant type at 21–39% in our results).

**LLM calls:** 4 + up to 2×N (debug)

**Inspired by:**
- [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128) — Chen et al., ICLR 2024. Introduces rubber duck debugging and execution-based debugging for LLMs.
- [LDB: A Large Language Model Debugger via Verifying Runtime Execution Step-by-step](https://arxiv.org/abs/2402.16906) — Zhong et al., ACL Findings 2024. Segments code into basic blocks and verifies intermediate values. Improves baseline by up to +9.8% on HumanEval/MBPP.

```bash
uv run inspect eval ./he_rel/_registry.py@humanevalrel \
    --model openrouter/openai/gpt-4.1-mini:nitro \
    --solver self_debugging
```

---

## V2 — AlphaCodium-style flow

**Solver:** `alphacodium`
**Files:** `alphacodium.py`, `alphacodium.yaml`

Restructured pipeline in two phases, inspired by AlphaCodium:

### Phase A — Preprocessing (natural-language reasoning, no code)

1. **Self-reflection** — The LLM reformulates the problem, identifies potential pitfalls, edge cases, and domain conventions (e.g., "reliability = 1 - CDF", "hazard rate ≠ probability")
2. **Solution candidates** — The LLM proposes 2–3 different approaches (analytic vs numeric, `scipy` vs manual implementation), ranked by robustness
3. **Test generation** — The LLM generates 3–5 additional test cases (edge cases, intermediate cases verifiable by hand). Key AlphaCodium idea: generating correct tests is easier than generating correct code

### Phase B — Iterative code (generation + execution + correction)

4. **Initial generation** — Code based on approach #1, using accumulated context
5. **Debug public tests** — Run against the docstring examples, retry on failure
6. **Debug AI tests** — Run against AI-generated tests, with **test anchoring**: if a correction breaks a public test that previously passed, reject the correction and revert to the previous version
7. **Loop** — Iterate until max M=4 total iterations

**LLM calls:** ~4 preprocessing + up to 2×M (iteration)

**Inspired by:**
- [Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering](https://arxiv.org/abs/2401.08500) — Ridnik et al., 2024. On CodeContests, improves GPT-4 from 19% to 44% pass@5. Introduces test anchors, double validation, and exploration-before-decision. [Source code](https://github.com/Codium-ai/AlphaCodium).
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — Shinn et al., NeurIPS 2023. Reaches 91% pass@1 on HumanEval (vs 80% for GPT-4 direct). The agent thinks aloud about failures and maintains an episodic memory.
- [MapCoder: Multi-Agent Code Generation for Competitive Problem Solving](https://arxiv.org/abs/2405.11403) — Islam et al., ACL 2024. Four LLM agents emulate the human cycle: recall similar examples, plan, generate, debug. SOTA on HumanEval (93.9%), MBPP (83.1%).

```bash
uv run inspect eval ./he_rel/_registry.py@humanevalrel \
    --model openrouter/openai/gpt-4.1-mini:nitro \
    --solver alphacodium
```

---

## Comparison

|                        | V0                | V1                           | V2                                      |
|------------------------|-------------------|------------------------------|-----------------------------------------|
| **LLM calls**          | 4 (fixed)         | 4 + up to 2×N (debug)        | ~4 preprocessing + up to 2×M (iter)     |
| **Code execution**     | No                | Yes (docstring tests)        | Yes (docstring tests + AI tests)        |
| **Feedback loop**      | No                | Yes (error → correction)     | Yes (with test anchors)                 |
| **Test generation**    | No                | No                           | Yes                                     |
| **Exploration**        | No                | No                           | Yes (2–3 solution candidates)           |

---

## Running all experiments

```bash
# All versions with gpt-4.1-mini
./scripts/run_experiments.sh

# Specific versions
./scripts/run_experiments.sh v0 v1 v2
```

---

## Additional avenues to explore

### 1. CodeChain — Modular self-revision
[CodeChain: Towards Modular Code Generation Through Chain of Self-revisions](https://arxiv.org/abs/2310.08992) — Le et al., ICLR 2024.

Decomposes solutions into abstract submodules, then extracts and clusters the best submodules across multiple generations for reuse. +35% pass@1 on APPS, +76% on CodeContests. Relevant for HumanEval-Rel because many reliability functions share common subroutines (numerical integration, CDF calculation, optimization).

### 2. LDB — Block-level debugging with intermediate values
[LDB: A Large Language Model Debugger](https://arxiv.org/abs/2402.16906) — Zhong et al., ACL Findings 2024.

Instead of showing the full traceback, segment the code into blocks and display intermediate variable values after each block. The LLM can then precisely locate where the computation diverges. Particularly useful for "wrong answer" errors where the code runs without exception but produces incorrect numeric results.

### 3. RAG with reliability documentation

Our paper identifies an "implementation gap": LLMs fail because there are no high-level libraries for reliability calculations. Injecting textbook excerpts (Rausand, Barlow & Proschan, Nakagawa) with exact formulas and reference code snippets via RAG would directly target the 9 questions never solved.

See: [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Gao et al., 2024.
