# Self-Specializing Stem Agent

A self-specializing agent system that starts as a domain-agnostic "stem" and gradually specializes itself through repeated task episodes and evidence-based evolution. The agent processes batches of tasks, evaluates its own outputs, proposes updates to its internal specialization state, and refines its behavior over time.

The LLM backend is pluggable: it can run against the **OpenAI API** (default, `gpt-4o-mini`) or against any **local Ollama model** of your choice. You switch by editing a single constant in `agents/llm_client.py`.

---

## Requirements

- Python 3.10+
- One of the following backends:
  - **OpenAI**: an `OPENAI_API_KEY` with access to the chosen chat model (default `gpt-4o-mini`), or
  - **Ollama**: [Ollama](https://ollama.com/) installed and running locally with a chat model pulled (e.g. `gemma3`, `llama3.1`, `qwen2.5`, …).

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
```

### Choosing a backend

Open `agents/llm_client.py` and edit the constants at the top:

```python
BACKEND = "openai"          # "openai" or "ollama"

OLLAMA_MODEL = "gemma3"     # any model you have pulled in Ollama
OPENAI_MODEL = "gpt-4o-mini"
```

#### Option A — OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

If the key is missing the agent fails fast at import time.

#### Option B — Ollama

```bash
# Start the local server (skip if already running)
ollama serve

# Pull whatever model you set in OLLAMA_MODEL
ollama pull gemma3            # or: ollama pull llama3.1, qwen2.5, ...
```

Any chat-capable Ollama model works — set `OLLAMA_MODEL` to its tag and you're done. There is no Gemma3-specific code path.

---

## Running the Main Loop

From inside the `stem_agent/` directory:

```bash
python controller/main_loop.py
```

This processes all 30 sample emails in `data/emails/`, running the full operational → evaluation → evolution cycle. Episodes are grouped into batches of 3 (`BATCH_SIZE` in `controller/main_loop.py`); each completed batch triggers one evolution step. The final evolved `specialization_state.yaml` is printed at the end.

---

## Running the Before/After Benchmark

After running the main loop at least once:

```bash
python evaluation/before_after.py
```

This compares stem agent performance using a blank (pre-evolution) state vs. the final evolved state on the same 30 emails. Results are printed as a comparison table and saved to `evaluation/benchmark_results.json`.

The benchmark uses the same backend you configured in `agents/llm_client.py`, so make sure your API key / Ollama server is still set up.

---

## Component Overview

| Component | Description |
|---|---|
| `agents/llm_client.py` | Single `call_llm()` helper. Switches between OpenAI and Ollama via the `BACKEND` constant; model name is configurable per backend |
| `agents/stem_agent.py` | Stem agent: `run_operational()`, `run_evolution()`, `apply_approved_changes()`. Loads the cumulative domain-hint history into the evolution prompt |
| `agents/evaluation_agent.py` | Evaluation agent: `evaluate_output()`, `evaluate_evolution()`, `check_maturity()` |
| `prompts/` | Four plain-text prompt files loaded at runtime; `{{placeholder}}` injection |
| `state/specialization_state.yaml` | Mutable agent self-model: domain, goals, workflow, action space, quality criteria, maturity stage |
| `state/evolution_log.jsonl` | Append-only log of all evolution proposals and their outcomes |
| `buffers/experience_buffer.json` | Accumulates 3 episode feedbacks per batch before triggering evolution |
| `buffers/domain_hints_log.jsonl` | Append-only log of every per-episode `domain_hint` ever observed; passed into each evolution prompt as `all_domain_hints_so_far` |
| `controller/main_loop.py` | Orchestration: operational loop → batch evaluation → evolution loop → maturity check |
| `data/emails/` | 30 sample email files covering diverse triage scenarios |
| `evaluation/before_after.py` | Standalone benchmark comparing pre- and post-evolution performance |

---

## How It Works

1. **Operational mode**: The stem agent processes each email using its current `specialization_state.yaml` as context. Each operational output also carries an `inferred_domain_hint` for that single task; non-null hints are appended to `buffers/domain_hints_log.jsonl`. The evaluation agent then scores the output per episode.

2. **Evolution mode**: Every 3 episodes, the stem agent reviews the batch of evaluated episodes — together with the **full cumulative hint history** — and proposes minimal, evidence-based updates to its specialization state. The evaluation agent approves, rejects, or requests one revision of the proposal. Approved changes are written back to the YAML.

3. **Domain inference**: `inferred_domain` is updated via a two-step process inside the stem agent prompt — derive a concise umbrella label from the current batch, then reconcile it against `all_domain_hints_so_far` so the label converges over time toward a short category covering everything seen so far.

4. **Maturity**: After 3 consecutive batches with no approved evolution (clean batches) and a non-null `inferred_domain`, the agent is marked as `mature` and subsequent episodes skip evaluation and evolution entirely.

5. **Maturity stage transitions**:
   - `exploring` → `specializing`: when `inferred_domain` is first set
   - `specializing` → `mature`: when the controller observes 3 consecutive clean batches
   - Only the controller sets `maturity_stage`; the stem agent never changes it directly
