# MAS-PromptBeach-Pro

A version of MAS-PromptBench with resumable benchmark runs, reusable model settings, and fixes for dataset loading and scoring.

This project is based on the original benchmark by Juyang Bai and Laixi Shi. See the [upstream repository](https://github.com/juyangbai/MAS-PromptBench), the [original paper](https://arxiv.org/abs/2606.23664), and the [MIT license](LICENSE).

[中文改动说明与使用方法](docs/benchmark_reliability.md) · [Documentation](docs/content/index.md)

<details open>
<summary><b>Contents</b></summary>

1. [Changes in MAS-PromptBeach-Pro](#changes-in-mas-promptbeach-pro)
2. [News and Updates](#-news-and-updates)
3. [Introduction](#-introduction)
4. [Code Structure](#-code-structure)
5. [Quickstart](#-quickstart)
6. [How to Extend](#-how-to-extend)
7. [Referenced Resources](#-referenced-resources)
8. [Citation](#-citation)

</details>

---

## Changes in MAS-PromptBeach-Pro

The main changes are in the centralized LangGraph runners and the shared API-Bank and ToolHop paths. Other runners can reuse the shared settings and dataset loaders.

- Save generated answers, code, patches, and token usage before scoring on supported paths. Use `--resume` to continue a run and `--retry-execution-errors` to retry execution failures. Result records distinguish these failures from ordinary wrong answers.
- Reuse model settings without storing API keys. Adapt known request fields for providers such as Qwen and DeepSeek, and check dataset dependencies before GPQA runs.
- Use HTTPS and a fixed User-Agent for HotpotQA's Wikipedia requests. In GPQA and BFCL, return an error `ToolMessage` if tool arguments cannot be parsed or a completed call has no reply.
- Fix function-based LiveCodeBench judging by reading the method name from metadata, parsing one argument per line, providing `typing` names such as `List`, and sending large inputs over standard input. APPS prefers local data and can fetch the test split's `test.jsonl` from a pinned revision.
- Reuse local SWE-bench repositories and container images, cache reference-patch checks, and use the official scoring library for project-specific test commands and log parsing. The centralized runner supports up to two evaluation workers.

### New files

| File | Purpose |
| --- | --- |
| [`core/checkpoints.py`](core/checkpoints.py) | Save generated artifacts and classify failures so interrupted runs can continue. |
| [`core/data.py`](core/data.py) | Look for prepared datasets and Hugging Face caches before downloading. |
| [`core/preflight.py`](core/preflight.py) | Check compatibility of `numpy`, `pyarrow`, and `datasets`. |
| [`core/profiles.py`](core/profiles.py) | Save and load model settings, including the last successful configuration for a model and endpoint. |
| [`core/providers.py`](core/providers.py) | Adapt request fields such as output token limits and thinking flags to known endpoints. |
| [`core/swe_harness.py`](core/swe_harness.py) | Run SWE-bench's official test and scoring logic in cached Singularity images. |
| [`core/swe_prepare.py`](core/swe_prepare.py) | Prepare SWE-bench repositories and images ahead of evaluation. |
| [`docs/benchmark_reliability.md`](docs/benchmark_reliability.md) | Explain configuration, caching, resume options, and task fixes in Chinese. |
| [`tests/core/test_benchmark_reliability.py`](tests/core/test_benchmark_reliability.py) | Cover request adaptation, checkpoints, and execution failure handling. |
| [`tests/core/test_dataset_and_judge_regressions.py`](tests/core/test_dataset_and_judge_regressions.py) | Cover local dataset reuse and APPS and LiveCodeBench judging. |
| [`tests/core/test_successful_profiles.py`](tests/core/test_successful_profiles.py) | Check that successful settings can be reused and are preserved after failed trials. |
| [`tests/core/test_swe_reliability.py`](tests/core/test_swe_reliability.py) | Cover repository reuse, reference checks, and scoring failure handling. |

See the [benchmark reliability guide](docs/benchmark_reliability.md) for setup and usage.

---

## 🔔 News and Updates

Updates from the original MAS-PromptBench project:

- **[2026-10-22]** — Supports eight prompt optimizers under a unified run protocol, with a codebase refactored onto a shared core.
- **[2026-06-22]** — Initial public release of the code!

---

## 📖 Introduction

<div align="center">
  <img src="https://juyangbai.github.io/MAS-PromptBench/assets/MAS-PromptBench_overview.png" alt="MAS-PromptBench overview" width="820">
</div>

MAS-PromptBench measures when prompt optimization improves multi-agent LLM systems. It runs real multi-agent runners on nine reasoning, coding, and tool-use datasets and optimizes their per-role prompts with eight methods under one run protocol.

- **Optimizer** — GEPA, MIPRO, MAPRO, MASPO, HiveMind, MAMUT-GEPA, MASPOB, and TAVO.
- **Task dataset** — GPQA-Diamond, HotpotQA, MATH, LiveCodeBench, APPS, BFCL, SWE-bench Verified, API-Bank, and ToolHop.
- **Workflow Topology** — `single`, `independent`, `sequential`, `centralized`, and `decentralized`, implemented across LangGraph, CrewAI, AutoGen, and the OpenAI Agents SDK.
- **Communication format** — three inter-agent message formats (`freeform`, `semi_structured`, `structured_soft`).
- **Team size** — the number of agents per team, `r ∈ {2, 4, 8, 10}`.
- **Task model** — `Qwen/Qwen3.5-9B`, plus `meta-llama/Llama-3.1-8B-Instruct` for a subset of cells.

---

## 🌳 Code Structure

| Path                                 | Contents                                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| [`assets/`](assets/)                 | logo and overview images                                                                                            |
| [`benchmarks/`](benchmarks/)         | evaluation ids, optimization splits, and the API-Bank source                                                        |
| [`communications/`](communications/) | inter-agent communication-format variants                                                                           |
| [`configs/`](configs/)               | seed role prompts (`prompts/<topology>/<dataset>/<role>.txt`) and team specs                                        |
| [`core/`](core/)                     | shared runner code and one task module per dataset                                                                  |
| [`docs/`](docs/)                     | source of the [documentation site](docs/content/index.md): tutorials, configuration and CLI reference               |
| [`models/`](models/)                 | vLLM serve scripts (Qwen3.5-9B / 122B, Llama-3.1-8B)                                                                |
| [`optimizers/`](optimizers/)         | eight prompt optimizers, their shared run protocol, and the real-runner bridge                                      |
| [`scripts/`](scripts/)               | sweep launchers                                                                                                     |
| [`teamsizes/`](teamsizes/)           | team-size variants (number of agents per team)                                                                      |
| [`tests/`](tests/)                   | unit tests and golden behavior snapshots                                                                            |
| [`topologies/`](topologies/)         | the core benchmark — 5 topologies × 9 datasets, one runnable pair each                                              |

---

## 🚀 Quickstart

Go from a fresh clone to scored results in four steps — install, serve a model, run a baseline, then optimize its prompts.

### 1. Install

Clone this repository using GitHub's **Code** button, then run the commands below from the cloned directory.

```bash
conda env create -f environment.yml      # Python 3.11 + vLLM + benchmark and optimizer deps
conda activate mas-promptbench
pip install --target vendor/openai_agents -r requirements-openai-agents.txt   # OpenAI Agents SDK (needs openai>=3, kept isolated)
```

`environment.yml` pins LangGraph, CrewAI, and AutoGen to the commits the results were produced with. Run every module with `python -m` from the repository root.

### 2. Serve a model

Set `VLLM_BASE_URL` and `MODEL_ID` to an **OpenAI-compatible** chat endpoint and model. For a hosted endpoint such as Qwen or DeepSeek, set your own `OPENAI_API_KEY` and follow the [model configuration guide](docs/benchmark_reliability.md#1-保存和复用模型配置). To serve a model locally, use a script from [`models/`](models/):

```bash
bash models/serve_qwen3_5_9b.sh
export VLLM_BASE_URL=http://localhost:8000/v1
export MODEL_ID=Qwen/Qwen3.5-9B
```

| Script                                                  | Model                                           | Serving                           | GPUs needed                                 |
| ------------------------------------------------------- | ----------------------------------------------- | --------------------------------- | ------------------------------------------- |
| [`serve_qwen3_5_9b.sh`](models/serve_qwen3_5_9b.sh)     | `Qwen/Qwen3.5-9B` (task model)                  | one replica per GPU, ports 8000+  | **≥ 1** CUDA GPU                            |
| [`serve_llama3_1_8b.sh`](models/serve_llama3_1_8b.sh)   | `meta-llama/Llama-3.1-8B-Instruct` (task model) | one replica per GPU, ports 8100+  | **≥ 1** CUDA GPU                            |
| [`serve_qwen3_5_122b.sh`](models/serve_qwen3_5_122b.sh) | `Qwen/Qwen3.5-122B-A10B-FP8` (reflection model) | tensor-parallel (TP=4), port 8200 | **4** FP8-capable GPUs (Hopper / Blackwell) |

Llama is gated: export `HF_TOKEN`, or serve a downloaded copy with `HF_HUB_OFFLINE=1` (see [model serving](docs/content/reference/environment.md#model-serving)).

### 3. Run a baseline

Every `(topology, dataset)` pair is a module with a smoke demo and a `--batch` mode that writes predictions and per-instance records under `results/`:

```bash
python -m topologies.single.hotpotqa.langgraph_hotpotqa                       # smoke demo
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100   # real batch on a slice
bash scripts/run_topologies.sh                                                # full sweep on the evaluation ids
```

See [topologies/README.md](topologies/README.md) for the run interface and per-dataset setup; [teamsizes/](teamsizes/README.md) and [communications/](communications/README.md) have their own sweeps.

### 4. Optimize prompts

[`optimizers/`](optimizers/) holds eight prompt optimizers — **GEPA**, **MIPRO**, **MAPRO**, **MASPO**, **HiveMind**, **MAMUT-GEPA**, **MASPOB**, and **TAVO** — that improve the seed prompts by running the **real** topology runners under one run protocol: 600 rollouts per job, deployment only if the optimized prompts beat the seeds on validation, and scoring on a held-out test split.

```bash
export TASK_ENDPOINTS=http://localhost:8000/v1             # task model
export REFLECTION_MODEL_BASE_URL=http://localhost:8200/v1  # reflection model

python -m optimizers.protocol.run --method gepa --dataset math --topology centralized \
    --model qwen --seed 0 --out runs/gepa/math/centralized/qwen/0
python -m optimizers.protocol.aggregate runs/ --out runs/summary.json   # paired summary over seeds 0-2
```

See [optimizers/README.md](optimizers/README.md).

---

## 🧩 How to Extend

Every piece is written once and reused, so an extension adds only what is new:

| Add           | What you write                                                                                                      | Guide                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Dataset**   | a task module in `core/tasks/`, a team spec, thin runners, seed prompts, evaluation IDs and splits, a bridge loader | [Add a Dataset](https://juyangbai.github.io/MAS-PromptBench/docs/extending/add-dataset/)      |
| **Topology**  | one thin runner per dataset on top of `core/`, a team spec and prompts, bridge adapters                             | [Add a Topology](https://juyangbai.github.io/MAS-PromptBench/docs/extending/add-topology/)    |
| **Optimizer** | a package `optimizers/<method>/` whose `integration.py` is registered with the run protocol                         | [Add an Optimizer](https://juyangbai.github.io/MAS-PromptBench/docs/extending/add-optimizer/) |

Then run the [tests](tests/golden/README.md): refactors must leave the golden snapshots unchanged.

---

## 🔗 Referenced Resources

MAS-PromptBench builds on the agent frameworks and libraries below, which retain their own upstream licenses, and evaluates on nine existing benchmarks. Please cite and comply with the license of each original dataset when reporting results.

<table>
  <tr><th>Category</th><th>Resource</th><th>Description</th></tr>
  <tr><td rowspan="5"><b>Agent framework</b></td><td><a href="https://github.com/langchain-ai/langgraph">LangGraph</a></td><td>stateful graph-based agent orchestration</td></tr>
  <tr><td><a href="https://github.com/crewAIInc/crewAI">CrewAI</a></td><td>role-based multi-agent framework</td></tr>
  <tr><td><a href="https://github.com/microsoft/autogen">AutoGen</a></td><td>conversational multi-agent framework</td></tr>
  <tr><td><a href="https://github.com/openai/openai-agents-python">OpenAI Agents SDK</a></td><td>agent runtime of the decentralized debate runners</td></tr>
  <tr><td><a href="https://github.com/composable-models/llm_multiagent_debate">LLM Multi-Agent Debate</a></td><td>multi-agent debate reference implementation</td></tr>
  <tr><td rowspan="9"><b>Benchmark</b></td><td><a href="https://github.com/idavidrein/gpqa">GPQA</a></td><td>graduate-level science multiple-choice QA</td></tr>
  <tr><td><a href="https://hotpotqa.github.io/">HotpotQA</a></td><td>multi-hop open-domain QA</td></tr>
  <tr><td><a href="https://github.com/hendrycks/math">MATH</a></td><td>competition mathematics</td></tr>
  <tr><td><a href="https://livecodebench.github.io/">LiveCodeBench</a></td><td>contamination-free code generation</td></tr>
  <tr><td><a href="https://github.com/hendrycks/apps">APPS</a></td><td>programming problems</td></tr>
  <tr><td><a href="https://gorilla.cs.berkeley.edu/leaderboard.html">BFCL</a></td><td>function / tool calling (Berkeley Function Calling Leaderboard)</td></tr>
  <tr><td><a href="https://www.swebench.com/">SWE-bench Verified</a></td><td>real-world GitHub issue resolution</td></tr>
  <tr><td><a href="https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank">API-Bank</a></td><td>tool-augmented API calling</td></tr>
  <tr><td><a href="https://huggingface.co/datasets/bytedance-research/ToolHop">ToolHop</a></td><td>multi-hop tool use</td></tr>
  <tr><td rowspan="2"><b>Library</b></td><td><a href="https://github.com/vllm-project/vllm">vLLM</a></td><td>model serving (the <code>models/</code> endpoints)</td></tr>
  <tr><td><a href="https://github.com/stanfordnlp/dspy">DSPy</a></td><td>the optimization backend of GEPA and MIPRO</td></tr>
</table>

---

## 📚 Citation

If you use MAS-PromptBeach-Pro in your research, please cite the original MAS-PromptBench paper:

```bibtex
@article{bai2026mas,
  title={MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent LLM Systems?},
  author={Bai, Juyang and Shi, Laixi},
  journal={arXiv preprint arXiv:2606.23664},
  year={2026}
}
```
