"""
Before/after benchmark script.
Compares stem agent performance with a blank specialization state vs
the final evolved state.

Run from the stem_agent/ directory:
    python evaluation/before_after.py
"""
import json
import logging
import sys
import uuid
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.stem_agent import StemAgent
from agents.evaluation_agent import EvaluationAgent

logging.basicConfig(
    level=logging.WARNING,  # Quiet during benchmark
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("before_after")

STATE_PATH = _ROOT / "state" / "specialization_state.yaml"
EMAILS_DIR = _ROOT / "data" / "emails"

# ---------------------------------------------------------------------------
# Blank state (all nulls / empty — same schema as initial state)
# ---------------------------------------------------------------------------
BLANK_STATE: dict = {
    "inferred_domain": None,
    "domain_confidence": 0.0,
    "goals": [],
    "input_model": {},
    "workflow_hypothesis": [],
    "action_space": [],
    "output_policy": {},
    "quality_criteria": [],
    "failure_modes": [],
    "open_questions": [],
    "maturity_stage": "exploring",
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(feedback_list: list) -> dict:
    total = len(feedback_list)
    if total == 0:
        return {"successful": 0.0, "partially_successful": 0.0, "unsuccessful": 0.0, "total": 0}

    counts = {"successful": 0, "partially_successful": 0, "unsuccessful": 0}
    for fb in feedback_list:
        result = fb.get("overall_result", "unsuccessful")
        if result in counts:
            counts[result] += 1

    return {
        "successful": round(counts["successful"] / total * 100, 1),
        "partially_successful": round(counts["partially_successful"] / total * 100, 1),
        "unsuccessful": round(counts["unsuccessful"] / total * 100, 1),
        "total": total,
    }


# ---------------------------------------------------------------------------
# Run one pass
# ---------------------------------------------------------------------------

def run_pass(
    label: str,
    email_paths: list,
    stem: StemAgent,
    evaluator: EvaluationAgent,
    state_override: dict,
) -> list:
    """
    Run all emails through the stem agent and evaluator using the given state.
    Returns a list of per-episode feedback dicts.
    """
    print(f"\nRunning {label} pass ({len(email_paths)} emails)...")
    feedbacks = []

    for i, path in enumerate(email_paths, 1):
        task_input = path.read_text(encoding="utf-8")
        episode_id = f"{label}-ep-{i:04d}"

        print(f"  [{i:2d}/{len(email_paths)}] {path.name}", end="", flush=True)

        output = stem.run_operational(task_input, state_override=state_override)
        feedback = evaluator.evaluate_output(
            episode_id=episode_id,
            specialization_state=state_override,
            task_input=task_input,
            stem_agent_output=output,
        )
        feedbacks.append(feedback)

        result = feedback.get("overall_result", "unknown")
        print(f"  → {result}")

    return feedbacks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stem = StemAgent()
    evaluator = EvaluationAgent()

    email_paths = sorted(EMAILS_DIR.glob("*.txt"))
    if not email_paths:
        print(f"ERROR: No email files found in {EMAILS_DIR}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load evolved state (after)
    # ------------------------------------------------------------------
    with open(STATE_PATH) as fh:
        evolved_state = yaml.safe_load(fh)

    if evolved_state.get("maturity_stage") == "exploring":
        print(
            "WARNING: The specialization state appears to be in 'exploring' stage.\n"
            "Run main_loop.py first to evolve the state before running this benchmark.\n"
            "Proceeding anyway — the 'after' results will use the current state as-is.\n"
        )

    # ------------------------------------------------------------------
    # BEFORE pass — blank state
    # ------------------------------------------------------------------
    before_feedbacks = run_pass(
        "BEFORE", email_paths, stem, evaluator, state_override=BLANK_STATE
    )

    # ------------------------------------------------------------------
    # AFTER pass — evolved state
    # ------------------------------------------------------------------
    after_feedbacks = run_pass(
        "AFTER", email_paths, stem, evaluator, state_override=evolved_state
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    before_metrics = compute_metrics(before_feedbacks)
    after_metrics = compute_metrics(after_feedbacks)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 52)
    print("           BEFORE vs AFTER COMPARISON")
    print("=" * 52)
    print(f"{'Metric':<28} | {'Before':>8} | {'After':>8}")
    print(f"{'-'*28}-+-{'-'*8}-+-{'-'*8}")
    print(
        f"{'Successful':<28} | "
        f"{before_metrics['successful']:>6.1f}% | "
        f"{after_metrics['successful']:>6.1f}%"
    )
    print(
        f"{'Partially Successful':<28} | "
        f"{before_metrics['partially_successful']:>6.1f}% | "
        f"{after_metrics['partially_successful']:>6.1f}%"
    )
    print(
        f"{'Unsuccessful':<28} | "
        f"{before_metrics['unsuccessful']:>6.1f}% | "
        f"{after_metrics['unsuccessful']:>6.1f}%"
    )
    print(f"{'-'*28}-+-{'-'*8}-+-{'-'*8}")
    print(f"{'Total episodes':<28} | {before_metrics['total']:>8} | {after_metrics['total']:>8}")
    print("=" * 52)

    # ------------------------------------------------------------------
    # Per-email breakdown
    # ------------------------------------------------------------------
    print("\nPer-email breakdown:")
    print(f"{'Email file':<40} | {'Before':<20} | {'After':<20}")
    print(f"{'-'*40}-+-{'-'*20}-+-{'-'*20}")
    for path, bf, af in zip(email_paths, before_feedbacks, after_feedbacks):
        b_result = bf.get("overall_result", "?")
        a_result = af.get("overall_result", "?")
        print(f"{path.name:<40} | {b_result:<20} | {a_result:<20}")

    print()

    # ------------------------------------------------------------------
    # Save raw results alongside the script
    # ------------------------------------------------------------------
    results_path = Path(__file__).parent / "benchmark_results.json"
    with open(results_path, "w") as fh:
        json.dump(
            {
                "before": {"metrics": before_metrics, "episodes": before_feedbacks},
                "after": {"metrics": after_metrics, "episodes": after_feedbacks},
            },
            fh,
            indent=2,
        )
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
