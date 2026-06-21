"""
graphs/state.py — State shared across all LangGraph graphs.

LangGraph passes this state between nodes. Each node returns a partial
dict with only the fields it changed — the graph merges it automatically.

CORRESPONDENCE WITH THE ARCHITECTURE DESCRIPTION (PDAS / PDAS_F):
  The three components of the session state are:
    (i)   conversation context   → "context"
    (ii)  form answer store      → "form_answers"
    (iii) execution pointer      → "current_stage_index"
  For PDAS_F, two additional fields persist across sessions:
    "feedback"      → serialized output of the Evaluation Agent
    "previous_plan" → serialized plan from the previous run

NOTE ON TOKEN COUNTERS:
  total_input_tokens and total_output_tokens use Annotated[int, operator.add]
  as their reducer. This means nodes must return ONLY the delta (tokens
  for this call), not the accumulated total. LangGraph sums them
  automatically. Correct example in a node:

      return {
          "total_input_tokens":  result.input_tokens,   # delta, not total
          "total_output_tokens": result.output_tokens,  # delta, not total
      }
"""

from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


# ---------------------------------------------------------------------------
# Main run state
# ---------------------------------------------------------------------------
class FormFillingState(TypedDict, total=False):
    # --- Conversation context ---
    context: list[dict]          # [{"who": "CHATBOT"|"CITIZEN", "message": "..."}]

    # --- Form ---
    form: dict
    form_name: str
    form_answers: dict           # {stage_id: raw_answer_text}
    filled_form: dict            # form JSON with "value" fields filled in

    # --- Form identification ---
    # Identifies which form the user wants to fill in. This is the
    # Planning Agent's first invocation of a session (there is no separate
    # Intent Agent — see agents.PDAS.call_planning_agent_identify_form).
    form_id: str | None          # "1"…"5" or "UNDEFINED" or None

    # --- Plan (planning-centric) ---
    plan: list[dict]
    plan_str: str
    current_stage_index: int
    single_task_output: str

    # --- Inter-run feedback (PDAS_F) ---
    feedback: str
    previous_plan: str

    # --- Counters and flags ---
    form_id_attempts: int
    validation_attempts: int
    iteration_count: int

    # --- Task Agent output validation (Validation Agent's dual invocation) ---
    # Counts replanning attempts for the current stage after a
    # NOT_VALIDATED on the SingleTaskAgent's output. Reset to 0 whenever
    # the run advances to a new stage (see node_store_answer).
    output_validation_attempts: int

    # --- Final results ---
    hit_rate:           float
    report:             dict
    advanced_metrics:   dict

    # --- Logging ---
    # Append-only accumulator of agent IO
    agent_io_log: Annotated[list[dict], operator.add]

    # Accumulated errors
    errors: Annotated[list[str], operator.add]

    # Explicit failure states (escalated for manual resolution), recorded
    # when a validation/replanning loop exceeds the maximum number of
    # attempts configured in RunConfig.
    escalations: Annotated[list[dict], operator.add]

    # Token counts
    total_input_tokens:     Annotated[int, operator.add]
    total_output_tokens:    Annotated[int, operator.add]
    total_tiktoken_input:   Annotated[int, operator.add]
    total_tiktoken_output:  Annotated[int, operator.add]

    # --- Simulated user profile ---
    user_profile: str | None


# ---------------------------------------------------------------------------
# Empty initial state
# ---------------------------------------------------------------------------
def initial_state() -> FormFillingState:
    return FormFillingState(
        context              = [],
        form                 = {},
        form_name            = "",
        form_answers         = {},
        filled_form          = {},
        form_id              = None,
        plan                 = [],
        plan_str             = "",
        current_stage_index  = 0,
        single_task_output   = "",
        feedback             = "",
        previous_plan        = "",
        form_id_attempts     = 0,
        validation_attempts  = 0,
        iteration_count      = 0,
        output_validation_attempts = 0,
        hit_rate             = 0.0,
        report               = {},
        advanced_metrics     = {},
        agent_io_log         = [],
        errors               = [],
        escalations          = [],
        total_input_tokens   = 0,
        total_output_tokens  = 0,
        user_profile         = None,
    )
