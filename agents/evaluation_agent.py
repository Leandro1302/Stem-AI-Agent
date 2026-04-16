"""
Evaluation agent: output evaluation and evolution evaluation.
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from agents.llm_client import call_llm

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = _ROOT / "prompts"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _inject(template: str, variables: dict) -> str:
    """Replace {{key}} placeholders in a template string."""
    for key, value in variables.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def _extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from an LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found (first 300 chars): {text[:300]}")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class EvaluationAgent:
    """Evaluates stem agent outputs and proposed specialization updates."""

    # ------------------------------------------------------------------
    # Output evaluation (per episode)
    # ------------------------------------------------------------------

    def evaluate_output(
        self,
        episode_id: str,
        specialization_state: dict,
        task_input: str,
        stem_agent_output: dict,
    ) -> dict:
        """
        Evaluate a single episode against the current specialization_state.
        Returns per-episode feedback JSON.
        """
        system_prompt = _load_prompt("evaluation_agent_system.txt")
        template = _load_prompt("output_evaluator_prompt.txt")

        user_prompt = _inject(
            template,
            {
                "episode_id": episode_id,
                "specialization_state": yaml.dump(
                    specialization_state, default_flow_style=False
                ),
                "task_input": task_input,
                "stem_agent_output": json.dumps(stem_agent_output, indent=2),
            },
        )

        response = call_llm(system_prompt, user_prompt)

        try:
            result = _extract_json(response)
        except Exception as exc:
            logger.warning(
                "Output evaluation JSON parse failed (%s); returning safe default.", exc
            )
            result = {
                "episode_id": episode_id,
                "task_input_summary": task_input[:120],
                "task_understanding": {"judgment": "partial", "notes": "JSON parse failed"},
                "workflow_step_feedback": [],
                "action_feedback": {
                    "chosen_action": "unknown",
                    "judgment": "not_assessable",
                    "notes": "JSON parse failed",
                },
                "output_feedback": {
                    "judgment": "acceptable",
                    "notes": "JSON parse failed; cannot assess",
                },
                "quality_feedback": [],
                "observed_failure_modes": ["evaluation_parse_failure"],
                "learning_signals": [],
                "overall_result": "partially_successful",
                "suggest_evolution_review": False,
            }

        # Always stamp the episode_id in case the model omitted it
        result["episode_id"] = episode_id
        return result

    # ------------------------------------------------------------------
    # Evolution evaluation (per batch)
    # ------------------------------------------------------------------

    def evaluate_evolution(
        self,
        specialization_state: dict,
        batch_buffer: dict,
        proposed_update: dict,
        force_binary: bool = False,
    ) -> dict:
        """
        Evaluate a proposed specialization update.
        Returns the evolution decision JSON.

        force_binary: when True (second/final attempt after a 'revise'),
        appends an instruction forbidding the 'revise' option so the model
        must return only 'approve' or 'reject'.
        """
        system_prompt = _load_prompt("evaluation_agent_system.txt")
        template = _load_prompt("evolution_evaluator_prompt.txt")

        user_prompt = _inject(
            template,
            {
                "specialization_state": yaml.dump(
                    specialization_state, default_flow_style=False
                ),
                "batch_buffer": json.dumps(batch_buffer, indent=2),
                "proposed_update": json.dumps(proposed_update, indent=2),
            },
        )

        if force_binary:
            user_prompt += (
                '\n\nIMPORTANT: This is the final evaluation. You may only respond with '
                '"approve" or "reject". The "revise" option is not available for this call.'
            )

        response = call_llm(system_prompt, user_prompt)

        try:
            result = _extract_json(response)
        except Exception as exc:
            logger.warning(
                "Evolution evaluation JSON parse failed (%s); defaulting to reject.", exc
            )
            result = {
                "decision": "reject",
                "rationale": "JSON parse failed; cannot assess proposal.",
                "proposal_assessment": {
                    "alignment_with_buffer": "low",
                    "expected_usefulness": "low",
                    "risk_of_overfitting": "high",
                    "state_consistency": "low",
                },
                "accepted_changes": [],
                "rejected_changes": ["all — parse failure"],
                "revision_guidance": [],
            }

        return result

    # ------------------------------------------------------------------
    # Maturity check
    # ------------------------------------------------------------------

    @staticmethod
    def check_maturity(consecutive_clean_batches: int) -> bool:
        """
        Return True when the agent has shown sustained stability,
        signaling the controller to set maturity_stage = 'mature'.
        Threshold: 3 consecutive clean batches.

        Note: the caller must also verify that inferred_domain is not null
        before acting on this — see the maturity guard in main_loop.py.
        """
        return consecutive_clean_batches >= 3
