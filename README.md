# qwen3.6-35b-a3b-rft

Rejection sampling fine-tuning (RFT) of **Qwen3.6-35B-A3B** for math, physics, tool calling and agentic Python,
on a Google Colab compute-unit budget, ending in a GGUF that runs locally in LM Studio.

The model generates several solutions to problems with **verifiable answers**, graders keep the correct ones, and the
**shortest correct solution** per problem becomes training data. A bf16 LoRA is trained on that data, evaluated against
the base model under identical conditions, and exported as an imatrix-calibrated mixed Q4 GGUF.

## Models

Both models are public on Hugging Face as mixed Q4 GGUFs (~21 GiB) with their imatrix files.

| Model | Training data |
|---|---|
| [Qwen3.6-35B-A3B-RFT-Agent-GGUF](https://huggingface.co/salihyesil59/Qwen3.6-35B-A3B-RFT-Agent-GGUF) | Round-1 RFT data plus multi-turn agent episodes (Python tool, sandboxed coding). **Recommended.** |
| [Qwen3.6-35B-A3B-RFT-Reasoning-GGUF](https://huggingface.co/salihyesil59/Qwen3.6-35B-A3B-RFT-Reasoning-GGUF) | Round-1 RFT data only: math, physics, Python and tool calls |

Results (FP8 on vLLM, 4 samples per problem, 16k tokens, truncated answers count as wrong):

| Set | Base | RFT-Reasoning | RFT-Agent |
|---|---|---|---|
| personal (4 problems) | 87.5% | 100% | 100% |
| MATH-500 (100-problem subset) | 70.0% | 73.0% | 73.5% |
| AIME 2025 | 12.5% | 12.5% | 13.3% |
| GPQA Diamond (physics, 86) | 55.2% | 63.4% | 62.5% |

With a Python tool (2 samples, 8k tokens per turn):

| Set | Base | RFT-Agent |
|---|---|---|
| MATH-500 + Python | 61.0% | 71.0% |
| AIME 2025 + Python | 3.3% | 6.7% |
| GPQA physics + Python | 79.1% | 79.1% |
| Agentic coding (100 held-out KodCode tasks) | 91.5% | 91.5% |

Most of the gain comes from shorter reasoning, i.e. fewer answers cut off at the token limit. Recommended settings:
thinking mode, temperature 0.6, top_p 0.95, top_k 20; offload the MoE experts to the CPU in LM Studio / llama.cpp.

## Pipeline

```
01 pilot (Qwen3.5-4B)  ─ validates data → LoRA → GGUF → LM Studio → eval end to end
        │
02 RFT generation      ─ baseline eval + single-turn solutions (math, physics, code, tool calls)   ─┐
03 Agent RFT           ─ tools baseline eval + multi-turn episodes (Python tool, sandboxed coding)  ─┤
        │                                                                                             │
04 SFT + eval          ─ bf16 LoRA on the shortest correct solutions; base vs RFT comparison  ◄──────┘
        │
05 GGUF                ─ merge → bf16 GGUF → imatrix → mixed Q4 → Hugging Face → LM Studio
```

| Notebook | What it does | Runtime | Cost |
|---|---|---|---|
| [01_pilot_sft](notebooks/01_pilot_sft.ipynb) | Pipeline smoke test on Qwen3.5-4B with Mixture-of-Thoughts data. Not about scores, only that every step runs. | L4 | 5–10 CU |
| [02_rft_generate](notebooks/02_rft_generate.ipynb) | **A)** baseline eval of the FP8 model with vLLM. **B)** samples 4 solutions per problem from a verifiable pool and grades them. Resumable. | A100 80GB | ~34 CU (5 h cap) |
| [03_agent_rft](notebooks/03_agent_rft.ipynb) | Multi-turn episodes where tool calls are **actually executed**: math/physics with `run_python`, and KodCode tasks solved with `read_file` / `write_file` / `run_tests`. Includes a held-out set of 100 agentic coding tasks. | A100 80GB | ~30–35 CU |
| [04_sft](notebooks/04_sft.ipynb) | Selects training data from 02 + 03, trains a bf16 LoRA with Unsloth, merges it, then re-runs both evals with vLLM and compares against the base model. | A100 80GB | ~20–30 CU |
| [05_gguf](notebooks/05_gguf.ipynb) | Merges the LoRA, converts to GGUF, computes an imatrix on domain text and produces a mixed Q4 (~22–23 GB), uploaded to Hugging Face. | A100 80GB | ~10 CU |

## Training data

Every source has an automatic grader; problems that appear in an eval set are removed from the pool.

| Source | Dataset | Grader |
|---|---|---|
| math | [`agentica-org/DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) (AIME ≤2023, AMC, Omni-MATH) | `\boxed{}` match |
| physics | [`TIGER-Lab/WebInstruct-verified`](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified), numeric answers | 2% relative tolerance |
| code | [`KodCode/KodCode-V1`](https://huggingface.co/datasets/KodCode/KodCode-V1), medium/hard | pytest |
| tool | [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (gated) | all calls + arguments, exact |
| agent | math/physics with a Python tool; KodCode in a sandbox | `\boxed{}` / tests pass after restoring the test file |

**Selection rules**
- Only correct and *complete* solutions (no truncation) are used, shortest first; this also shortens the reasoning.
- Partially solved problems (1/4–3/4) carry the real learning signal: up to 2 solutions each.
- Problems solved 4/4 are kept at a reduced rate, one solution each; never-solved problems are saved for a later round.
- Examples are rendered with the model's **own chat template** (`reasoning_content` → `<think>`, `tool_calls` → XML) and
  skipped, not truncated, when they exceed the training length (10240 tokens).

## Evaluation

All comparisons use the same settings for base and fine-tuned model: FP8 weights on vLLM, 4 samples per task,
temperature 0.6, 16k max tokens. Both result files are re-scored with the current graders before comparison.
**Gate:** if the RFT model does not beat the base model, it is not used.

| Set | Source |
|---|---|
| `personal` | a small hand-written set in [eval/tasks/personal.jsonl](eval/tasks/personal.jsonl) (Python, math, tool calls) |
| `math500` | 100 problems from MATH-500 |
| `aime2025` | AIME 2025 |
| `gpqa_physics` | GPQA Diamond, physics subset |
| `*+py`, `agent_code` | the same sets with a Python tool, plus 100 held-out agentic coding tasks |

## Quantization

The 35B-A3B MoE is meant to run locally with the experts offloaded to the CPU, so the quant is mixed:

| Tensors | Type | Why |
|---|---|---|
| Experts (`ffn_*_exps`) | Q4_K_M | ~90% of the size; kept on the CPU |
| Attention, linear attention, shared expert | Q8_0 | small but sensitive; on the GPU |
| Output layer / embedding | Q8_0 / Q6_K | |

The imatrix is computed on ~300 of the model's own RFT solutions (math, physics, code, tool calls, agent episodes) plus
Turkish and English Wikipedia articles.

## Local eval harness

[`eval/`](eval/) runs against any OpenAI-compatible server (LM Studio by default). The Colab notebooks import the same
graders, so local and Colab results use the same format.

| File | Purpose |
|---|---|
| [run_eval.py](eval/run_eval.py) | Runs task files and grades them: `math`, `numeric`, `mcq`, `python`, `pytest`, `tool` |
| [agent.py](eval/agent.py) | Multi-turn tool-use episodes (Python tool, sandboxed coding) |
| [fetch_benchmarks.py](eval/fetch_benchmarks.py) | Downloads MATH-500, AIME 2025 and GPQA physics into `eval/tasks/` |
| [rescore.py](eval/rescore.py) | Re-grades a saved results file without running the model |

```bash
pip install requests datasets pytest
```

```bash
python eval/fetch_benchmarks.py
```

```bash
python eval/run_eval.py --tasks eval/tasks/personal.jsonl eval/tasks/math500.jsonl --label base
```

Results go to `eval/results/` (per-sample JSONL + `summary.csv`).

> **Warning:** the agent tools are not a real sandbox: model-written code runs on the host machine.

## Running it on Colab

1. Copy `eval/` to Google Drive as `MyDrive/colab-ai/eval/`.
2. Add an `HF_TOKEN` (write access) in Colab's **Secrets** panel and enable notebook access.
3. Accept the terms of the gated datasets on Hugging Face: [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) and
   [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).
4. Run the notebooks in order. Every long step writes to Drive and resumes when the session drops.
5. In 04 and 05, use *Runtime → Restart session* between the Unsloth and vLLM/llama.cpp parts, **not**
   "Disconnect and delete runtime", which deletes the merged model on local disk.

## Notes

- GPQA questions must not be published; downloaded benchmark files and results are git-ignored.
- KodCode is licensed CC BY-NC 4.0, which applies to the code portion of the training data.

## License

Code in this repository: [Apache 2.0](LICENSE). Datasets and base model keep their own licenses.
