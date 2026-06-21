"""
agents/PDAS.py — Agents exclusive to the PDAS and PDAS_F architectures.

Includes: EvaluationAgent, PlanningAgent (plan generation, form
identification, feedback-driven planning, and in-session replanning), and
SingleTaskAgent.

All prompts are read from prompts/ (flat directory, shared by both
architectures).
"""

from __future__ import annotations

import json

from agents.base import AgentResult, run_agent
from config import RunConfig


def call_planning_agent_identify_form(
    context: list[dict[str, str]],
    run_cfg: RunConfig,
    use_full_context: bool = True,
) -> AgentResult:
    """
    First Planning Agent invocation of a session: identifies which form the
    user wants to fill in from their message, before any plan exists. This
    subsumes what used to be a dedicated Intent Agent — there is no
    separate intent-detection entity in the architecture.

    Returns {"form_id": "1".."5"} or {"form_id": "UNDEFINED"}.

    Args:
        context:          Message history [{"who": ..., "message": ...}, ...].
        run_cfg:          Run configuration (model, architecture, etc.).
        use_full_context: If True, sends the full history; if False, only the last message.
    """
    if use_full_context:
        text = "\n".join(f"{m['who']}: {m['message']}" for m in context)
    else:
        last = context[-1]
        text = f"{last['who']}: {last['message']}"

    return run_agent(
        prompt_file  = "PlanningAgentIdentifyForm.txt",
        replacements = {"[Inserir 1]": text},
        run_cfg      = run_cfg,
        agent_name   = "planning_agent",
    )


def call_evaluation_agent(
    filled_form: dict,
    planning_output: str,
    hit_rate: float,
    iteration_count: int,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Analyzes the filled form vs. the original plan and generates structured
    feedback (metrics + advice) for the next run's PlanningAgent.

    The hit_rate and iteration_count metrics are pre-computed by the code
    and passed directly into the prompt to avoid the model computing them
    incorrectly from the JSON.

    Returns JSON with: metricas, erros_por_campo, manter, ajustar, mudar.

    Args:
        filled_form:     Filled form (dict with "secoes" and "campos" with "value").
        planning_output: Original plan, serialized (JSON string from the PlanningAgent).
        hit_rate:        Rate of correctly filled required fields (0–1).
        iteration_count: Number of iterations/stages executed in this run.
        run_cfg:         Run configuration.
    """
    
    metrics_str = json.dumps(
        {"hit_rate": round(hit_rate, 4), "iteration_count": iteration_count},
        ensure_ascii=False,
    )
    return run_agent(
        prompt_file  = "EvaluationAgent.txt",
        replacements = {
            "[Inserir 1]": filled_form,
            "[Inserir 2]": planning_output,
            "[Inserir 3]": metrics_str,
            "[Inserir threshold]": str(run_cfg.hit_rate_threshold),
        },
        run_cfg      = run_cfg,
        agent_name   = "evaluation_agent",
    )


def call_planning_agent(
    form: dict,
    run_cfg: RunConfig,
    condition_map: dict,
) -> AgentResult:
    """
    Generates the strategic plan of stages to fill in a form.
    PDAS architecture (no inter-session feedback).

    Returns {"plan": [{"stage": N, "action": ..., "input": ..., "output": ...}, ...]}.

    Args:
        form:          Loaded form JSON.
        run_cfg:       Run configuration.
        condition_map: Field visibility conditions (from utils.forms.preprocess_conditions).
    """
    return run_agent(
        prompt_file  = "PlanningAgent.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir condições]": json.dumps(condition_map, ensure_ascii=False),
        },
        run_cfg      = run_cfg,
        agent_name   = "planning_agent",
    )


def call_planning_agent_with_feedback(
    form: dict,
    feedback: str,
    previous_plan: str,
    run_cfg: RunConfig,
    condition_map: dict,
) -> AgentResult:
    """
    Generates the strategic plan incorporating the previous run's
    EvaluationAgent feedback. PDAS_F architecture (with feedback).

    Returns {"plan": [{"stage": N, "action": ..., "input": ..., "output": ...}, ...]}.

    Args:
        form:          Loaded form JSON.
        feedback:      Serialized EvaluationAgent output (empty string on the first run).
        previous_plan: Previous run's plan (empty string on the first run).
        run_cfg:       Run configuration.
        condition_map: Field visibility conditions (from utils.forms.preprocess_conditions).
    """
    return run_agent(
        prompt_file  = "PlanningAgentWithFeedback.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir 2]": feedback      or "(sem feedback — primeira execução)",
            "[Inserir 3]": previous_plan or "(sem plano anterior — primeira execução)",
            "[Inserir condições]": json.dumps(condition_map, ensure_ascii=False),
            "[Inserir threshold]": str(run_cfg.hit_rate_threshold),
        },
        run_cfg      = run_cfg,
        agent_name   = "planning_agent",
    )


def call_single_task_agent(
    task_action: str,
    task_input: str,
    task_expected_output: str,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Executes a single stage of the plan generated by the PlanningAgent.
    Returns {"output": "..."}.

    Args:
        task_action:          "action" field of the plan stage.
        task_input:           Serialized input for this stage (JSON string).
        task_expected_output: "output" field of the plan stage (description of the expected output).
        run_cfg:              Run configuration.
    """
    return run_agent(
        prompt_file  = "SingleTaskAgent.txt",
        replacements = {
            "[Inserir 1]": task_action,
            "[Inserir 2]": task_input,
            "[Inserir 3]": task_expected_output,
        },
        run_cfg      = run_cfg,
        agent_name   = "single_task_agent",
    )


def call_planning_agent_revise(
    form: dict,
    current_plan: list[dict],
    current_stage_index: int,
    validation_reason: str,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Within-session replanning, triggered when the Validation Agent returns
    NOT_VALIDATED for a Task Agent's output (see
    graphs/shared_nodes.node_validate_output and node_replan_stage).

    Unlike call_planning_agent_with_feedback (which incorporates the
    Evaluation Agent's feedback across sessions), this call revises the
    current session's plan mid-execution, preserving the stages already
    completed and adjusting only the current stage and/or pending stages.

    Returns {"plan": [{"stage": N, "action": ..., "input": ..., "output": ...}, ...]}.
    Strict preservation of the already-completed stages (index <
    current_stage_index) is also enforced in code by node_replan_stage,
    regardless of what the model returns.

    Args:
        form:                 Loaded form JSON.
        current_plan:         Full current plan (list of stages).
        current_stage_index:  Index (0-based) of the stage whose output was rejected.
        validation_reason:    "reason" returned by the Validation Agent on NOT_VALIDATED.
        run_cfg:              Run configuration.
    """
    return run_agent(
        prompt_file  = "ReplanningAgent.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir 2]": current_plan,
            "[Inserir índice atual]": str(current_stage_index),
            "[Inserir 3]": validation_reason,
        },
        run_cfg      = run_cfg,
        agent_name   = "planning_agent",
    )
