"""
Main orchestration loop for the self-specializing stem agent system.
Run from the stem_agent/ directory:
    python controller/main_loop.py
"""
import json
import logging
import sys
import uuid
from pathlib import Path

import yaml

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.stem_agent import StemAgent
from agents.evaluation_agent import EvaluationAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_loop")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATE_PATH = _ROOT / "state" / "specialization_state.yaml"
EVOLUTION_LOG_PATH = _ROOT / "state" / "evolution_log.jsonl"
BUFFER_PATH = _ROOT / "buffers" / "experience_buffer.json"
DOMAIN_HINTS_LOG_PATH = _ROOT / "buffers" / "domain_hints_log.jsonl"
EMAILS_DIR = _ROOT / "data" / "emails"

BATCH_SIZE = 3


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    with open(STATE_PATH) as fh:
        return yaml.safe_load(fh)


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as fh:
        yaml.dump(state, fh, default_flow_style=False, allow_unicode=True)


def set_maturity_stage(stage: str) -> None:
    state = load_state()
    state["maturity_stage"] = stage
    save_state(state)
    logger.info("maturity_stage set to '%s'", stage)


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def _blank_buffer(batch_id: str) -> dict:
    return {
        "batch_id": batch_id,
        "batch_size": 0,
        "episodes": [],
        "summary": None,
    }


def compute_batch_summary(episodes: list) -> dict:
    summary = {
        "overall_results": {
            "successful": 0,
            "partially_successful": 0,
            "unsuccessful": 0,
        },
        "task_understanding_issues": 0,
        "workflow_step_issue_counts": {},
        "action_issue_count": 0,
        "output_issue_count": 0,
        "quality_issue_counts": {},
        "observed_failure_modes": [],
        "recurrent_learning_signals": [],
        "domain_hints": [],
    }

    learning_signal_counts: dict = {}

    for ep in episodes:
        result = ep.get("overall_result", "unsuccessful")
        if result in summary["overall_results"]:
            summary["overall_results"][result] += 1

        understanding = ep.get("task_understanding", {}).get("judgment", "")
        if understanding in ("partial", "incorrect"):
            summary["task_understanding_issues"] += 1

        for step_fb in ep.get("workflow_step_feedback", []):
            step = step_fb.get("step", "unknown")
            if step_fb.get("judgment") in ("partial", "incorrect"):
                summary["workflow_step_issue_counts"][step] = (
                    summary["workflow_step_issue_counts"].get(step, 0) + 1
                )

        action_judgment = ep.get("action_feedback", {}).get("judgment", "")
        if action_judgment in ("partially_appropriate", "inappropriate"):
            summary["action_issue_count"] += 1

        output_judgment = ep.get("output_feedback", {}).get("judgment", "")
        if output_judgment in ("acceptable", "poor"):
            summary["output_issue_count"] += 1

        for qf in ep.get("quality_feedback", []):
            criterion = qf.get("criterion", "unknown")
            if qf.get("judgment") in ("partially_met", "not_met"):
                summary["quality_issue_counts"][criterion] = (
                    summary["quality_issue_counts"].get(criterion, 0) + 1
                )

        for fm in ep.get("observed_failure_modes", []):
            if fm not in summary["observed_failure_modes"]:
                summary["observed_failure_modes"].append(fm)

        for ls in ep.get("learning_signals", []):
            learning_signal_counts[ls] = learning_signal_counts.get(ls, 0) + 1

        hint = ep.get("domain_hint")
        if hint:
            summary["domain_hints"].append(hint)

    # Only include learning signals seen more than once (recurrent)
    summary["recurrent_learning_signals"] = [
        ls for ls, count in learning_signal_counts.items() if count > 1
    ]
    # Fallback: include all if none are recurrent
    if not summary["recurrent_learning_signals"]:
        summary["recurrent_learning_signals"] = list(learning_signal_counts.keys())

    return summary


def save_buffer(buffer: dict) -> None:
    with open(BUFFER_PATH, "w") as fh:
        json.dump(buffer, fh, indent=2)


# ---------------------------------------------------------------------------
# Evolution log helper
# ---------------------------------------------------------------------------

def append_evolution_log(entry: dict) -> None:
    with open(EVOLUTION_LOG_PATH, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    stem = StemAgent()
    evaluator = EvaluationAgent()

    email_paths = sorted(EMAILS_DIR.glob("*.txt"))
    if not email_paths:
        logger.error("No email files found in %s", EMAILS_DIR)
        sys.exit(1)

    logger.info("Loaded %d email(s) from %s", len(email_paths), EMAILS_DIR)

    iteration = 0
    batch_counter = 0
    consecutive_clean_batches = 0

    current_batch_id = str(uuid.uuid4())
    buffer = _blank_buffer(current_batch_id)

    for email_path in email_paths:
        task_input = email_path.read_text(encoding="utf-8")
        iteration += 1
        episode_id = f"ep-{iteration:04d}"
        logger.info("--- Episode %s | file: %s ---", episode_id, email_path.name)

        state = load_state()

        # ----------------------------------------------------------------
        # Mature mode: run operationally only, no evaluation / evolution
        # ----------------------------------------------------------------
        if state.get("maturity_stage") == "mature":
            logger.info("[MATURE] Running operational only.")
            output = stem.run_operational(task_input)
            print(f"\n[{episode_id}] OUTPUT:\n{json.dumps(output, indent=2)}\n")
            continue

        # ----------------------------------------------------------------
        # Operational step
        # ----------------------------------------------------------------
        output = stem.run_operational(task_input)
        logger.info("[%s] Operational output: action=%s", episode_id, output.get("action"))

        # ----------------------------------------------------------------
        # Evaluate output
        # ----------------------------------------------------------------
        episode_feedback = evaluator.evaluate_output(
            episode_id=episode_id,
            specialization_state=state,
            task_input=task_input,
            stem_agent_output=output,
        )
        logger.info(
            "[%s] Evaluation: %s", episode_id, episode_feedback.get("overall_result")
        )

        # ----------------------------------------------------------------
        # Update maturity_stage: exploring → specializing
        # ----------------------------------------------------------------
        state = load_state()
        if (
            state.get("maturity_stage") == "exploring"
            and state.get("inferred_domain") is not None
        ):
            set_maturity_stage("specializing")

        # Also check the domain hint from the operational output
        elif (
            state.get("maturity_stage") == "exploring"
            and output.get("inferred_domain_hint")
        ):
            # The stem agent hinted at a domain but hasn't set it yet.
            # We store the hint only if the inferred_domain is still null;
            # actual domain setting happens through the evolution loop.
            logger.info(
                "[%s] Domain hint observed: %s", episode_id, output.get("inferred_domain_hint")
            )

        # ----------------------------------------------------------------
        # Append episode to buffer (with domain_hint from operational output)
        # ----------------------------------------------------------------
        episode_hint = output.get("inferred_domain_hint")
        episode_feedback["domain_hint"] = episode_hint
        buffer["episodes"].append(episode_feedback)
        buffer["batch_size"] = len(buffer["episodes"])

        # Persist non-null hint to the append-only history log
        if episode_hint:
            with open(DOMAIN_HINTS_LOG_PATH, "a") as fh:
                fh.write(json.dumps({"episode_id": episode_id, "domain_hint": episode_hint}) + "\n")

        # ----------------------------------------------------------------
        # Evolution loop when batch is full
        # ----------------------------------------------------------------
        if len(buffer["episodes"]) >= BATCH_SIZE:
            batch_counter += 1
            logger.info("=== Batch %d complete — running evolution loop ===", batch_counter)

            buffer["summary"] = compute_batch_summary(buffer["episodes"])
            save_buffer(buffer)

            state = load_state()

            # --- Stem agent proposes an update ---
            proposed_update = stem.run_evolution(buffer)
            logger.info(
                "Proposal type: %s | summary: %s",
                proposed_update.get("proposal_type"),
                proposed_update.get("summary", "")[:80],
            )

            if proposed_update.get("proposal_type") == "no_update":
                logger.info("No update proposed — incrementing clean batch counter.")
                consecutive_clean_batches += 1
                append_evolution_log(
                    {
                        "iteration": batch_counter,
                        "proposal": proposed_update.get("summary", "no_update"),
                        "reason": "proposal_type=no_update",
                        "status": "rejected",
                    }
                )

            else:
                # --- Evaluation agent assesses the proposal ---
                evolution_decision = evaluator.evaluate_evolution(
                    specialization_state=state,
                    batch_buffer=buffer,
                    proposed_update=proposed_update,
                )
                decision = evolution_decision.get("decision", "reject")
                logger.info("Evolution decision: %s", decision)

                if decision == "approve":
                    stem.apply_approved_changes(proposed_update["proposed_changes"])
                    append_evolution_log(
                        {
                            "iteration": batch_counter,
                            "proposal": proposed_update.get("summary", ""),
                            "reason": evolution_decision.get("rationale", ""),
                            "status": "approved",
                        }
                    )
                    consecutive_clean_batches = 0

                elif decision == "reject":
                    append_evolution_log(
                        {
                            "iteration": batch_counter,
                            "proposal": proposed_update.get("summary", ""),
                            "reason": evolution_decision.get("rationale", ""),
                            "status": "rejected",
                        }
                    )
                    consecutive_clean_batches += 1

                elif decision == "revise":
                    logger.info("Revision requested — running one revision attempt.")
                    revision_guidance = evolution_decision.get("revision_guidance", [])

                    revised_proposal = stem.run_evolution(
                        buffer, revision_guidance=revision_guidance
                    )

                    state = load_state()
                    final_decision = evaluator.evaluate_evolution(
                        specialization_state=state,
                        batch_buffer=buffer,
                        proposed_update=revised_proposal,
                        force_binary=True,
                    )
                    final_outcome = final_decision.get("decision", "reject")
                    logger.info("Final decision after revision: %s", final_outcome)

                    if final_outcome == "approve":
                        stem.apply_approved_changes(revised_proposal["proposed_changes"])
                        append_evolution_log(
                            {
                                "iteration": batch_counter,
                                "proposal": revised_proposal.get("summary", ""),
                                "reason": final_decision.get("rationale", ""),
                                "status": "approved",
                            }
                        )
                        consecutive_clean_batches = 0
                    else:
                        append_evolution_log(
                            {
                                "iteration": batch_counter,
                                "proposal": revised_proposal.get("summary", ""),
                                "reason": final_decision.get("rationale", ""),
                                "status": "rejected",
                            }
                        )
                        consecutive_clean_batches += 1

            # --- Maturity check ---
            # Guard: never promote to mature if inferred_domain is still null.
            # An empty specialization state has not converged — reset the counter.
            state = load_state()
            if state.get("inferred_domain") is None and consecutive_clean_batches > 0:
                logger.info(
                    "Maturity guard: inferred_domain is null — resetting "
                    "consecutive_clean_batches from %d to 0.",
                    consecutive_clean_batches,
                )
                consecutive_clean_batches = 0
            elif EvaluationAgent.check_maturity(consecutive_clean_batches):
                set_maturity_stage("mature")
                print("\nAgent has reached maturity. Evolution complete.\n")

            # --- Reset buffer for next batch ---
            current_batch_id = str(uuid.uuid4())
            buffer = _blank_buffer(current_batch_id)

    # ----------------------------------------------------------------
    # Final state printout
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SPECIALIZATION STATE")
    print("=" * 60)
    final_state = load_state()
    print(yaml.dump(final_state, default_flow_style=False, allow_unicode=True))


if __name__ == "__main__":
    main()
