# Self-Specializing Stem Agent

A self-specializing agent system that starts as a domain-agnostic "stem" and gradually specializes itself through repeated task episodes and evidence-based evolution. The agent processes batches of tasks, evaluates its own outputs, proposes updates to its internal specialization state, and refines its behavior over time — all driven by Gemma3 running locally via Ollama.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Gemma3 model pulled in Ollama (`ollama pull gemma3`)

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Ollama (if not already running)
ollama serve

# 3. Pull the Gemma3 model
ollama pull gemma3
```

---

## Running the Main Loop

From inside the `stem_agent/` directory:

```bash
python controller/main_loop.py
```

This processes all 15 sample emails in `data/emails/`, running the full operational → evaluation → evolution cycle. The final evolved `specialization_state.yaml` is printed at the end.

---

## Running the Before/After Benchmark

After running the main loop at least once:

```bash
python evaluation/before_after.py
```

This compares stem agent performance using a blank (pre-evolution) state vs the final evolved state on the same 15 emails. Results are printed as a comparison table and saved to `evaluation/benchmark_results.json`.

---

## Component Overview

| Component | Description |
|---|---|
| `agents/llm_client.py` | Single `call_llm()` helper for all Gemma3/Ollama calls |
| `agents/stem_agent.py` | Stem agent: `run_operational()`, `run_evolution()`, `apply_approved_changes()` |
| `agents/evaluation_agent.py` | Evaluation agent: `evaluate_output()`, `evaluate_evolution()`, `check_maturity()` |
| `prompts/` | Four plain-text prompt files loaded at runtime; `{{placeholder}}` injection |
| `state/specialization_state.yaml` | Mutable agent self-model: domain, goals, workflow, action space, quality criteria |
| `state/evolution_log.jsonl` | Append-only log of all evolution proposals and their outcomes |
| `buffers/experience_buffer.json` | Accumulates 3 episode feedbacks per batch before triggering evolution |
| `controller/main_loop.py` | Orchestration: operational loop → batch evaluation → evolution loop → maturity check |
| `data/emails/` | 15 sample email files covering diverse triage scenarios |
| `evaluation/before_after.py` | Standalone benchmark comparing pre- and post-evolution performance |

---

## How It Works

1. **Operational mode**: The stem agent processes each email using its current `specialization_state.yaml` as context. The evaluation agent scores the output per episode.

2. **Evolution mode**: Every 3 episodes, the stem agent reviews the batch of evaluated episodes and proposes minimal, evidence-based updates to its specialization state. The evaluation agent approves, rejects, or requests a revision of the proposal. Approved changes are written back to the YAML.

3. **Maturity**: After 2 consecutive batches with no approved evolution (clean batches), the agent is marked as `mature` and subsequent episodes skip evaluation and evolution entirely.

4. **Maturity stage transitions**:
   - `exploring` → `specializing`: when `inferred_domain` is first set
   - `specializing` → `mature`: when the evaluation agent signals maturity (2 clean batches)
   - Only the controller sets `maturity_stage`; the stem agent never changes it directly
