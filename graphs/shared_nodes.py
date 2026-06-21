"""
graphs/shared_nodes.py — LangGraph nodes reused across architectures.

Each node is a pure function: state + run_cfg + log_mgr → partial state dict.
The graph automatically merges the returned dict into the current state.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json

from config import RunConfig
from graphs.state import FormFillingState
from utils.forms import load_form, safe_json_loads, hit_rate_calculation, preprocess_conditions, evaluate_with_conditions, calculate_advanced_metrics
from utils.logging import RunLog, LogFileManager, color_print
from agents.base import _tok_multi
from agents.interface import call_interface_agent
from agents.validation import call_validation_agent, call_validation_agent_output
from agents.PDAS import call_planning_agent, call_planning_agent_with_feedback
from agents.PDAS import call_single_task_agent, call_planning_agent_revise
from agents.PDAS import call_planning_agent_identify_form
from agents.PDAS import call_evaluation_agent
from agents.auto_user import call_auto_user_agent, load_random_user_profile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _tok(result) -> dict:
    """
    Returns only this call's token delta.
    LangGraph sums it automatically via the operator.add reducer defined in state.py.
    NEVER accumulate manually — that would cause double counting.
    """
    return {
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
        "total_tiktoken_input": result.tiktoken_input,
        "total_tiktoken_output": result.tiktoken_output,
    }


def _log_entry(agent: str, inp: Any, out: str) -> dict:
    """Builds an entry for agent_io_log."""
    return {"agent": agent, "input": inp, "output": out}


def _fallback_fill_form(state: FormFillingState) -> dict:
    """
    Deterministic, code-only fallback used when the Validation Agent is on
    the plan's final stage but fails to return a well-formed
    "final_output" (e.g. malformed JSON), or when the run is cut short
    before reaching the real final stage (see make_route_stage_or_done's
    max_plan_stages safety net). No LLM call here — pure Python, so it
    never fails the run even if every agent call did.

    Strategy, in order:
      1. Map by exact field id (campo1, campo2...)
      2. Map by label similarity (case-insensitive)
      3. Use the conversation transcript to find a value for each field
    """
    import copy

    form    = state.get("form", {})
    answers = state.get("form_answers", {})
    context = state.get("context", [])
    context_str = "\n".join(
        f"{m['who']}: {m['message']}" for m in context
        if m["who"] in ("CITIZEN", "CHATBOT")
    )
    flat_answers = {**{str(k): v for k, v in answers.items()}, "conversation": context_str}

    filled = copy.deepcopy(form)

    def _find_value(campo: dict) -> str | None:
        cid   = campo.get("id",    "").lower()
        label = campo.get("label", "").lower()
        for key, val in flat_answers.items():
            if key.lower() in (cid, label) or label in key.lower() or cid in key.lower():
                return val
        return None

    for secao in filled.get("secoes", []):
        for campo in secao.get("campos", []):
            if campo.get("value") is None:
                val = _find_value(campo)
                if val:
                    campo["value"] = val

    return filled


# ---------------------------------------------------------------------------
# Node: initial greeting
# ---------------------------------------------------------------------------
def node_greet(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    msg = run_cfg.greeting_message
    ctx = [{"who": "CHATBOT", "message": msg}]
    log_mgr.write("chat", f"CHATBOT: {msg}\n")
    log_mgr.write("geral", "[INFO] Saudação inicial\n")
    color_print(msg, "CHATBOT")
    return {"context": ctx}


# ---------------------------------------------------------------------------
# Node: user input (human or simulated)
# ---------------------------------------------------------------------------
def node_get_user_input(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    context = state.get("context", [])

    if run_cfg.auto_user:
        profile = state.get("user_profile") or load_random_user_profile(
            seed=run_cfg.auto_user_seed
        )
        last_bot = context[-1]["message"] if context else "Início da conversa."
        result  = call_auto_user_agent(profile, last_bot, run_cfg)
        message = result.content
        color_print(message, "CITIZEN")
        tok = _tok(result)
    else:
        from config import Colors
        message = input(f"{Colors.BLUE}CIDADÃO: {Colors.RESET}")
        tok     = {}
        profile = state.get("user_profile")

    new_ctx = context + [{"who": "CITIZEN", "message": message}]
    log_mgr.write("chat", f"CITIZEN: {message}\n")
    log_mgr.write("geral", "[INFO] Mensagem do cidadão\n")

    return {
        "context":      new_ctx,
        "user_profile": profile,
        **tok,
    }


# ---------------------------------------------------------------------------
# Node: validation of the user's message
# ---------------------------------------------------------------------------
def node_validate(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    context = state.get("context", [])
    result  = call_validation_agent(context, run_cfg)
    parsed  = safe_json_loads(result.content) or {}
    status  = parsed.get("status", "NOT_VALIDATED")

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] validation_agent → {result.content}", "DEBUG")

    log_mgr.record("validation_agent", context[-1]["message"], result.content)

    attempts = state.get("validation_attempts", 0)
    if status != "VALIDATED":
        attempts += 1
    else:
        attempts = 0

    return {
        **_tok(result),
        "total_tiktoken_input":     result.tiktoken_input,
        "total_tiktoken_output":    result.tiktoken_output,
        "validation_attempts":      attempts,
        "agent_io_log":             [_log_entry("validation_agent", context[-1]["message"], result.content)],
    }


# ---------------------------------------------------------------------------
# Node: explicit escalation after exceeding max_validation_attempts (user input)
# ---------------------------------------------------------------------------
def node_escalate_validation(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """
    Explicit failure state (see architecture description): when the
    Interface Agent cannot obtain a valid response from the user within
    run_cfg.max_validation_attempts, the correction loop ends safely
    instead of continuing indefinitely — the stage is flagged for manual
    resolution and execution proceeds with the last response received.
    """
    idx     = state.get("current_stage_index", 0)
    context = state.get("context", [])
    last_msg = context[-1]["message"] if context else ""
    msg = (
        f"[ESCALATED] Etapa {idx}: utilizador excedeu "
        f"max_validation_attempts={run_cfg.max_validation_attempts} sem resposta válida. "
        f"A avançar com a última mensagem recebida ('{last_msg[:80]}') para resolução manual."
    )
    log_mgr.write("geral", f"[ERROR] {msg}\n")
    color_print(msg, "ERROR")

    return {
        "errors":      [msg],
        "escalations": [{
            "type":         "user_input_validation",
            "stage_index":  idx,
            "attempts":     state.get("validation_attempts", 0),
            "last_message": last_msg,
        }],
    }


# ---------------------------------------------------------------------------
# Node: form identification (Planning Agent's first invocation of a session)
# ---------------------------------------------------------------------------
def node_identify_form(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """
    Identifies which form the user wants to fill in. There is no separate
    Intent Agent in this architecture — this is simply the first thing the
    Planning Agent does in a session, before any plan exists (see
    agents.PDAS.call_planning_agent_identify_form).
    """
    context = state.get("context", [])
    result  = call_planning_agent_identify_form(context, run_cfg, run_cfg.use_full_context)
    parsed  = safe_json_loads(result.content) or {}
    form_id = parsed.get("form_id", "UNDEFINED")

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] planning_agent (identify form) → {result.content}", "DEBUG")

    log_mgr.record("planning_agent", context[-1]["message"], result.content)

    attempts = state.get("form_id_attempts", 0)
    if form_id == "UNDEFINED":
        attempts += 1
    else:
        attempts = 0

    return {
        **_tok(result),
        "total_tiktoken_input":     result.tiktoken_input,
        "total_tiktoken_output":    result.tiktoken_output,
        "form_id":                  form_id,
        "form_id_attempts":         attempts,
        "agent_io_log":             [_log_entry("planning_agent", context[-1]["message"], result.content)],
    }


# ---------------------------------------------------------------------------
# Node: response when the form could not be identified (via InterfaceAgent)
# ---------------------------------------------------------------------------
def node_form_unclear(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    context     = state.get("context", [])
    instruction = "Adapte a seguinte mensagem, mantendo o estilo do utilizador: Desculpe, não entendi. Pode reformular o que precisa?"
    result      = call_interface_agent(context, instruction, run_cfg, run_cfg.use_full_context)
    msg         = result.content

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] interface_agent (form_unclear) → {msg}", "DEBUG")

    color_print(msg, "CHATBOT")
    new_ctx = context + [{"who": "CHATBOT", "message": msg}]
    log_mgr.write("chat", f"CHATBOT: {msg}\n")
    log_mgr.record("interface_agent", instruction, msg)

    return {
        **_tok(result),
        "total_tiktoken_input":     result.tiktoken_input,
        "total_tiktoken_output":    result.tiktoken_output,
        "context":                  new_ctx,
        "agent_io_log":             [_log_entry("interface_agent", instruction, msg)],
    }


# ---------------------------------------------------------------------------
# Node: form loading
# ---------------------------------------------------------------------------
def node_load_form(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    form_id   = state.get("form_id", "1")
    form_name = f"formulario_{form_id}"
    form      = load_form(form_name)

    visible_fields, condition_map = preprocess_conditions(form)

    log_mgr.write("geral", f"[INFO] Formulário carregado: {form_name} | Campos visíveis incondicionais: {len(visible_fields)}\n")

    return {
        "form": form,
        "form_name": form_name,
        "visible_fields": visible_fields,
        "condition_map": condition_map,
    }


# ---------------------------------------------------------------------------
# Node: planning (no feedback)
# ---------------------------------------------------------------------------
def node_plan(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    form   = state.get("form", {})
    _, condition_map = preprocess_conditions(form)
    result = call_planning_agent(form, run_cfg, condition_map)
    parsed = safe_json_loads(result.content) or {}

    if isinstance(parsed, list):
        plan = parsed
    else:
        plan = parsed.get("plan", [])

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] planning_agent → {len(plan)} etapas", "DEBUG")

    log_mgr.record("planning_agent", form, result.content)
    errors = [] if plan else ["planning_agent devolveu plano vazio"]

    return {
        **_tok(result),
        "total_tiktoken_input":     result.tiktoken_input,
        "total_tiktoken_output":    result.tiktoken_output,
        "plan":                     plan,
        "plan_str":                 result.content,
        "errors":                   errors,
        "agent_io_log":             [_log_entry("planning_agent", "form", result.content)],
    }


# ---------------------------------------------------------------------------
# Node: planning with feedback
# ---------------------------------------------------------------------------
def node_plan_with_feedback(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    form          = state.get("form", {})
    _, condition_map = preprocess_conditions(form)
    feedback      = state.get("feedback", "")
    previous_plan = state.get("previous_plan", "")
    result        = call_planning_agent_with_feedback(form, feedback, previous_plan, run_cfg, condition_map)
    parsed        = safe_json_loads(result.content) or {}

    if isinstance(parsed, list):
        plan = parsed
    else:
        plan = parsed.get("plan", [])

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] planning_agent_with_feedback → {len(plan)} etapas", "DEBUG")

    log_mgr.record("planning_agent", form, result.content)
    errors = [] if plan else ["planning_agent_with_feedback devolveu plano vazio"]

    return {
        **_tok(result),
        "total_tiktoken_input":     result.tiktoken_input,
        "total_tiktoken_output":    result.tiktoken_output,
        "plan":                     plan,
        "plan_str":                 result.content,
        "errors":                   errors,
        "agent_io_log":             [_log_entry("planning_agent", "form+feedback", result.content)],
    }


# ---------------------------------------------------------------------------
# Node: execute one plan stage (Single Task Agent only)
# ---------------------------------------------------------------------------
def node_execute_single_task(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """
    Runs the SingleTaskAgent for the current plan stage. The output is NOT
    shown to the user yet — it first goes through the Validation Agent
    (node_validate_output) to confirm it satisfies the stage's expected
    output. Only after VALIDATED (or escalation) does the InterfaceAgent
    (node_send_stage_message) adapt the final message.
    """
    plan  = state.get("plan", [])
    idx   = state.get("current_stage_index", 0)
    form  = state.get("form", {})

    if idx >= len(plan):
        return {}

    stage      = plan[idx]
    task_input = json.dumps(state.get("single_task_output") or form, ensure_ascii=False)

    st_result = call_single_task_agent(
        task_action          = stage["action"],
        task_input           = task_input,
        task_expected_output = stage["output"],
        run_cfg              = run_cfg,
    )
    if run_cfg.debug_mode:
        color_print(f"[DEBUG] single_task_agent (stage {idx+1}) → {st_result.content[:80]}", "DEBUG")
    log_mgr.record("single_task_agent", stage, st_result.content)

    return {
        "single_task_output":       st_result.content,
        **_tok(st_result),
        "total_tiktoken_input":     st_result.tiktoken_input,
        "total_tiktoken_output":    st_result.tiktoken_output,
        "agent_io_log": [
            _log_entry("single_task_agent", stage, st_result.content),
        ],
    }


# ---------------------------------------------------------------------------
# Node: validate the Task Agent's output (2nd invocation of the Validation Agent)
# ---------------------------------------------------------------------------
def node_validate_output(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """
    The Validation Agent verifies both the user's responses (node_validate)
    and the Task Agents' outputs — this is the second invocation. It
    compares the output generated by the SingleTaskAgent against the
    stage's expected output. A NOT_VALIDATED acts as the replanning
    trigger (node_replan_stage); the number of attempts is capped by
    run_cfg.max_replanning_attempts.

    On the plan's final stage, if the output is VALIDATED, the response
    also carries "final_output" — the fully compiled form — which this
    node stores as state["filled_form"], taking over the role the Output
    Generator Agent used to play.
    """
    plan   = state.get("plan", [])
    idx    = state.get("current_stage_index", 0)
    output = state.get("single_task_output", "")

    if idx >= len(plan):
        return {}

    stage        = plan[idx]
    total_tasks  = len(plan)
    task_number  = idx + 1
    is_last_stage = task_number == total_tasks

    result = call_validation_agent_output(
        task_action          = stage.get("action", ""),
        task_input            = stage.get("input", ""),
        expected_output       = stage.get("output", ""),
        generated_output      = output,
        current_task_number   = task_number,
        total_tasks           = total_tasks,
        run_cfg               = run_cfg,
        original_form         = state.get("form", {}),
        form_answers          = state.get("form_answers", {}),
    )
    parsed = safe_json_loads(result.content) or {}
    status = parsed.get("status", "NOT_VALIDATED")

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] validation_agent (output, stage {task_number}/{total_tasks}) → {result.content[:120]}", "DEBUG")

    log_mgr.record("validation_agent", stage, result.content)

    attempts = state.get("output_validation_attempts", 0)
    if status != "VALIDATED":
        attempts += 1
    else:
        attempts = 0

    update = {
        **_tok(result),
        "total_tiktoken_input":       result.tiktoken_input,
        "total_tiktoken_output":      result.tiktoken_output,
        "output_validation_attempts": attempts,
        # "validation_agent_output" tag, distinct from "validation_agent",
        # so routing functions don't confuse this invocation with the
        # user-input one.
        "agent_io_log": [_log_entry("validation_agent_output", stage, result.content)],
    }

    if status == "VALIDATED" and is_last_stage:
        final_output = parsed.get("final_output")
        if not (isinstance(final_output, dict) and "secoes" in final_output):
            # Safety net: the model skipped or malformed final_output —
            # fall back to a deterministic merge instead of failing the run.
            final_output = _fallback_fill_form(state)
        update["filled_form"] = final_output

    return update


# ---------------------------------------------------------------------------
# Node: replan the current stage (Planning Agent receives validation feedback)
# ---------------------------------------------------------------------------
def node_replan_stage(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """
    Triggered when node_validate_output returns NOT_VALIDATED. The Planning
    Agent receives the invalidation reason and generates a revised plan,
    preserving the stages already completed (index < current_stage_index).

    Preservation of those stages is enforced in code, not just by the
    prompt: we always replace the prefix of the plan returned by the model
    with the original, already-completed stages, keeping only the revised
    part (current stage onward) from the model's response.
    """
    plan = state.get("plan", [])
    idx  = state.get("current_stage_index", 0)
    form = state.get("form", {})

    reason = ""
    for entry in reversed(state.get("agent_io_log", [])):
        if entry.get("agent") == "validation_agent_output":
            parsed = safe_json_loads(entry["output"]) or {}
            reason = parsed.get("reason", "")
            break

    result = call_planning_agent_revise(
        form                  = form,
        current_plan          = plan,
        current_stage_index   = idx,
        validation_reason      = reason,
        run_cfg               = run_cfg,
    )
    parsed = safe_json_loads(result.content) or {}
    model_plan = parsed if isinstance(parsed, list) else parsed.get("plan", [])

    if model_plan:
        # Code-level guarantee: already-completed stages are never altered,
        # regardless of what the model returns.
        completed   = plan[:idx]
        revised_pending = model_plan[idx:] if len(model_plan) > idx else model_plan
        new_plan    = completed + revised_pending
    else:
        new_plan = plan  # failed to generate a revised plan — keep the previous one

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] planning_agent (replan, stage {idx+1}) → {len(new_plan)} etapas", "DEBUG")

    log_mgr.record("planning_agent", {"stage_index": idx, "reason": reason}, result.content)

    return {
        **_tok(result),
        "total_tiktoken_input":  result.tiktoken_input,
        "total_tiktoken_output": result.tiktoken_output,
        "plan":                  new_plan,
        "plan_str":              result.content,
        "agent_io_log":          [_log_entry("planning_agent", {"stage_index": idx, "reason": reason}, result.content)],
    }


# ---------------------------------------------------------------------------
# Node: explicit escalation after exceeding max_replanning_attempts (Task Agent output)
# ---------------------------------------------------------------------------
def node_escalate_output_validation(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """
    Explicit failure state: if a stage's output remains NOT_VALIDATED
    after run_cfg.max_replanning_attempts Planning Agent revisions, the
    system does not keep replanning indefinitely — it records the failure
    and proceeds with the last output generated, flagging the stage for
    manual resolution.
    """
    idx = state.get("current_stage_index", 0)
    msg = (
        f"[ESCALATED] Etapa {idx}: output do Task Agent continuou NOT_VALIDATED "
        f"após max_replanning_attempts={run_cfg.max_replanning_attempts} revisões do plano. "
        f"A avançar com o último output gerado para resolução manual."
    )
    log_mgr.write("geral", f"[ERROR] {msg}\n")
    color_print(msg, "ERROR")

    return {
        "errors":      [msg],
        "escalations": [{
            "type":        "output_validation",
            "stage_index": idx,
            "attempts":    state.get("output_validation_attempts", 0),
        }],
    }


# ---------------------------------------------------------------------------
# Node: send the stage's message to the user (InterfaceAgent)
# ---------------------------------------------------------------------------
def node_send_stage_message(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """
    Adapts the Task Agent's output (already validated, or escalated after
    exhausting the replanning attempts) to the user's style and sends it
    as the chatbot's message.
    """
    ctx    = state.get("context", [])
    output = state.get("single_task_output", "")

    instruction = f"Adapte a seguinte mensagem, mantendo o estilo de linguagem do utilizador: {output}"
    if_result   = call_interface_agent(ctx, instruction, run_cfg, run_cfg.use_full_context)
    msg         = if_result.content

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] interface_agent → {msg[:80]}", "DEBUG")

    color_print(msg, "CHATBOT")
    new_ctx = ctx + [{"who": "CHATBOT", "message": msg}]
    log_mgr.write("chat", f"CHATBOT: {msg}\n")
    log_mgr.write("geral", "[INFO] Mensagem do chatbot\n")
    log_mgr.record("interface_agent", instruction, msg)

    return {
        "context":                  new_ctx,
        **_tok(if_result),
        "total_tiktoken_input":     if_result.tiktoken_input,
        "total_tiktoken_output":    if_result.tiktoken_output,
        "agent_io_log": [
            _log_entry("interface_agent", instruction, msg),
        ],
    }


# ---------------------------------------------------------------------------
# Node: store the user's answer for the current stage
# ---------------------------------------------------------------------------
def node_store_answer(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    plan    = state.get("plan", [])
    idx     = state.get("current_stage_index", 0)
    context = state.get("context", [])
    answers = dict(state.get("form_answers", {}))

    user_msg = context[-1]["message"] if context else ""
    if idx < len(plan):
        stage    = plan[idx]
        stage_id = stage.get("stage", idx) if isinstance(stage, dict) else idx
        answers[stage_id] = user_msg

    parciais = state.get("respostas_parciais", []) + [
        {"tarefa": (plan[idx].get("stage", idx) if isinstance(plan[idx], dict) else idx) if idx < len(plan) else idx, "resposta": user_msg}
    ]

    return {
        "form_answers":        answers,
        "current_stage_index": idx + 1,
        "iteration_count":     state.get("iteration_count", 0) + 1,
        "validation_attempts": 0,
        "output_validation_attempts": 0,
    }


# ---------------------------------------------------------------------------
# Node: metrics calculation (hit_rate)
# ---------------------------------------------------------------------------
def node_calculate_metrics(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    filled       = state.get("filled_form") or _fallback_fill_form(state)
    original     = state.get("form", {})
    answers      = state.get("form_answers", {})
    context      = state.get("context", [])

    #hr, rep = hit_rate_calculation(filled)
    hr, rep = evaluate_with_conditions(filled, original, answers)
    try:
        adv = calculate_advanced_metrics(state, original, filled, context)
    except Exception as e:
        print(f"[ERROR ADVANCED METRICS] {e}")
        adv = {"error": str(e)}

    return {"hit_rate": hr, "report": rep, "advanced_metrics": adv, "filled_form": filled}


# ---------------------------------------------------------------------------
# Node: evaluation agent
# ---------------------------------------------------------------------------
def node_evaluate(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    filled          = state.get("filled_form", {})
    plan_str        = state.get("plan_str", "")
    hit_rate        = state.get("hit_rate", 0.0)
    iteration_count = state.get("iteration_count", 0)

    result = call_evaluation_agent(filled, plan_str, hit_rate, iteration_count, run_cfg)

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] evaluation_agent → {result.content[:80]}", "DEBUG")

    log_mgr.record("evaluation_agent", plan_str, result.content)

    return {
        **_tok(result),
        "total_tiktoken_input": result.tiktoken_input,
        "total_tiktoken_output": result.tiktoken_output,
        "feedback":     result.content,
        "agent_io_log": [_log_entry("evaluation_agent", "plan_str", result.content)],
    }


# ---------------------------------------------------------------------------
# Routing functions (used in conditional_edges)
# ---------------------------------------------------------------------------
def make_route_after_validation(run_cfg: RunConfig):
    """
    Routing factory for validating the user's message.

    Uses run_cfg.max_validation_attempts instead of a hardcoded value.
    Once the limit is exceeded, returns "escalated" — the graph should
    connect this to node_escalate_validation before proceeding to
    store_answer, recording the explicit failure state instead of
    advancing silently.
    """
    def _route(state: FormFillingState) -> str:
        attempts = state.get("validation_attempts", 0)
        if attempts >= run_cfg.max_validation_attempts:
            return "escalated"
        logs = state.get("agent_io_log", [])
        for entry in reversed(logs):
            if entry.get("agent") == "validation_agent":
                parsed = safe_json_loads(entry["output"]) or {}
                if parsed.get("status") == "VALIDATED":
                    return "validated"
                return "not_validated"
        return "validated"  # fallback
    return _route


def make_route_after_output_validation(run_cfg: RunConfig):
    """
    Routing factory for validating the Task Agent's output (2nd invocation
    of the Validation Agent). A NOT_VALIDATED triggers replanning
    (node_replan_stage); once run_cfg.max_replanning_attempts is exceeded,
    returns "escalated" so the graph connects to
    node_escalate_output_validation before sending the stage's message to
    the user.
    """
    def _route(state: FormFillingState) -> str:
        attempts = state.get("output_validation_attempts", 0)
        if attempts >= run_cfg.max_replanning_attempts:
            return "escalated"
        logs = state.get("agent_io_log", [])
        for entry in reversed(logs):
            if entry.get("agent") == "validation_agent_output":
                parsed = safe_json_loads(entry["output"]) or {}
                if parsed.get("status") == "VALIDATED":
                    return "validated"
                return "not_validated"
        return "validated"  # fallback
    return _route


def make_route_after_form_identification(run_cfg: RunConfig):
    """
    Routing factory for the form-identification step (the Planning
    Agent's first invocation of a session). Uses
    run_cfg.max_form_identification_attempts instead of a hardcoded value.
    """
    def _route(state: FormFillingState) -> str:
        form_id  = state.get("form_id", "UNDEFINED")
        attempts = state.get("form_id_attempts", 0)
        if attempts >= run_cfg.max_form_identification_attempts:
            return "max_attempts"
        if form_id == "UNDEFINED":
            return "undefined"
        return "found"
    return _route


def make_route_stage_or_done(run_cfg: RunConfig):
    """
    Routing factory to check whether there are still plan stages left to execute.

    Returns a LangGraph-compatible callable that uses
    run_cfg.max_plan_stages instead of a hardcoded value. Use as:

        g.add_conditional_edges(
            "store_answer",
            make_route_stage_or_done(run_cfg),
            {"next_stage": "execute_single_task", "done": "metrics"},
        )
    """
    def _route(state: FormFillingState) -> str:
        idx  = state.get("current_stage_index", 0)
        plan = state.get("plan", [])
        if idx >= len(plan) or idx >= run_cfg.max_plan_stages:
            return "done"
        return "next_stage"
    return _route
