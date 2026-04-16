"""
Stem agent: operational and evolution modes.
"""
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import yaml

from agents.llm_client import call_llm

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
STATE_PATH = _ROOT / "state" / "specialization_state.yaml"
PROMPTS_DIR = _ROOT / "prompts"
DOMAIN_HINTS_LOG_PATH = _ROOT / "buffers" / "domain_hints_log.jsonl"


def _load_all_domain_hints() -> list:
    if not DOMAIN_HINTS_LOG_PATH.exists():
        return []
    hints = []
    with open(DOMAIN_HINTS_LOG_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                hint = entry.get("domain_hint")
                if hint:
                    hints.append(hint)
            except json.JSONDecodeError:
                continue
    return hints


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_state(state_override: Optional[dict] = None) -> dict:
    if state_override is not None:
        return state_override
    with open(STATE_PATH) as fh:
        return yaml.safe_load(fh)


def _save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as fh:
        yaml.dump(state, fh, default_flow_style=False, allow_unicode=True)


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from an LLM response."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block  ```json ... ```  or  ``` ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First { ... } in the response
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM response (first 300 chars): {text[:300]}")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class StemAgent:
    """Implements operational and evolution modes for the stem agent."""

    # ------------------------------------------------------------------
    # Operational mode
    # ------------------------------------------------------------------

    def run_operational(
        self,
        task_input: str,
        state_override: Optional[dict] = None,
    ) -> dict:
        """
        Process a single task instance using the current specialization state.

        Returns a dict with at minimum: action, output, reasoning.
        """
        state = _load_state(state_override)
        system_prompt = _load_prompt("stem_agent_system.txt")

        user_prompt = (
            "MODE: OPERATIONAL\n\n"
            "Current specialization_state:\n"
            f"{yaml.dump(state, default_flow_style=False)}\n\n"
            "Task input:\n"
            f"{task_input}\n\n"
            "Analyze this task using the current specialization state and produce your best output.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "action": "string — the action you are taking",\n'
            '  "output": "string — your full response / result for this task",\n'
            '  "reasoning": "string — brief explanation of your decisions",\n'
            '  "inferred_domain_hint": "string — your best guess at the domain for this specific task, regardless of whether one is already set in specialization_state"\n'
            "}"
        )

        response = call_llm(system_prompt, user_prompt)

        try:
            result = _extract_json(response)
        except Exception as exc:
            logger.warning("Operational output JSON parse failed (%s); wrapping raw text.", exc)
            result = {
                "action": "respond",
                "output": response,
                "reasoning": "JSON parse failed; raw output preserved.",
                "inferred_domain_hint": None,
            }

        return result

    # ------------------------------------------------------------------
    # Evolution mode
    # ------------------------------------------------------------------

    def run_evolution(
        self,
        batch_buffer: dict,
        revision_guidance: Optional[list] = None,
        state_override: Optional[dict] = None,
    ) -> dict:
        """
        Propose a minimal, evidence-based update to specialization_state.

        revision_guidance: list of strings from the evaluation agent when
        a "revise" decision was returned on the first attempt.
        """
        state = _load_state(state_override)
        system_prompt = _load_prompt("stem_agent_system.txt")

        revision_block = ""
        if revision_guidance:
            revision_block = (
                "\n\nREVISION GUIDANCE from evaluation agent "
                "(your previous proposal was marked 'revise'):\n"
                + json.dumps(revision_guidance, indent=2)
                + "\n\nPlease revise your proposal accordingly before responding."
            )

        schema = json.dumps(
            {
                "proposal_id": "string",
                "based_on_batch_id": "string",
                "proposal_type": "update | no_update",
                "summary": "string",
                "proposed_changes": [
                    {
                        "field": (
                            "inferred_domain | domain_confidence | goals | input_model | "
                            "workflow_hypothesis | action_space | output_policy | "
                            "quality_criteria | failure_modes | open_questions"
                        ),
                        "operation": "set | add | remove | revise",
                        "path": "string",
                        "current_value_summary": "string",
                        "proposed_value": {},
                        "justification": "string",
                        "supporting_evidence": ["string"],
                    }
                ],
                "expected_benefit": ["string"],
                "uncertainties": ["string"],
            },
            indent=2,
        )

        all_hints = _load_all_domain_hints()

        user_prompt = (
            "MODE: EVOLUTION\n\n"
            "Current specialization_state:\n"
            f"{yaml.dump(state, default_flow_style=False)}\n\n"
            "Completed batch buffer (3 evaluated episodes + summary):\n"
            f"{json.dumps(batch_buffer, indent=2)}\n\n"
            "all_domain_hints_so_far (every non-null domain_hint observed since the agent started, across all past batches):\n"
            f"{json.dumps(all_hints, indent=2)}\n"
            f"{revision_block}\n\n"
            "Analyze the batch evidence and propose a minimal, evidence-based update to "
            "specialization_state.\n"
            "If the evidence does not justify any update, set proposal_type to 'no_update' "
            "and leave proposed_changes as an empty list.\n"
            "Return ONLY valid JSON matching exactly this schema:\n\n"
            f"{schema}"
        )

        response = call_llm(system_prompt, user_prompt)

        try:
            result = _extract_json(response)
        except Exception as exc:
            logger.warning("Evolution proposal JSON parse failed (%s); defaulting to no_update.", exc)
            result = {
                "proposal_id": str(uuid.uuid4()),
                "based_on_batch_id": batch_buffer.get("batch_id", "unknown"),
                "proposal_type": "no_update",
                "summary": "JSON parse failed; treating as no_update.",
                "proposed_changes": [],
                "expected_benefit": [],
                "uncertainties": ["JSON parse failure on LLM response"],
            }

        # Ensure proposal_id is always set
        if not result.get("proposal_id"):
            result["proposal_id"] = str(uuid.uuid4())

        return result

    # ------------------------------------------------------------------
    # Apply approved changes
    # ------------------------------------------------------------------

    def apply_approved_changes(self, proposed_changes: list) -> None:
        """
        Read the current YAML, apply structured proposed_changes field by field,
        and write the file back.

        Each element in proposed_changes must have:
          field, operation (set|add|remove|revise), proposed_value
        """
        state = _load_state()

        for change in proposed_changes:
            field = change.get("field")
            operation = change.get("operation", "set")
            proposed_value = change.get("proposed_value")

            if field not in state:
                logger.warning("Unknown field '%s' in proposed_changes; skipping.", field)
                continue

            if operation == "set":
                state[field] = proposed_value

            elif operation == "revise":
                state[field] = proposed_value

            elif operation == "add":
                current = state[field]
                if isinstance(current, list):
                    if isinstance(proposed_value, list):
                        # Add items not already present
                        for item in proposed_value:
                            if item not in current:
                                current.append(item)
                    else:
                        if proposed_value not in current:
                            current.append(proposed_value)
                elif isinstance(current, dict) and isinstance(proposed_value, dict):
                    current.update(proposed_value)
                else:
                    state[field] = proposed_value

            elif operation == "remove":
                current = state[field]
                if isinstance(current, list):
                    if isinstance(proposed_value, list):
                        state[field] = [x for x in current if x not in proposed_value]
                    elif proposed_value in current:
                        current.remove(proposed_value)
                elif isinstance(current, dict) and isinstance(proposed_value, str):
                    current.pop(proposed_value, None)

            else:
                logger.warning("Unknown operation '%s' for field '%s'; skipping.", operation, field)

        _save_state(state)
        logger.info("Specialization state updated with %d change(s).", len(proposed_changes))
