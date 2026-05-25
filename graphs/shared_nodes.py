"""
graphs/shared_nodes.py — Nós LangGraph reutilizáveis entre arquitecturas.

Cada nó é uma função pura:  state + run_cfg + log_mgr → dict parcial de estado.
O grafo faz merge automático do dict retornado com o estado atual.
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
from agents.intent import call_intent_agent
from agents.interface import call_interface_agent
from agents.validation import call_validation_agent
from agents.PDAS import call_planning_agent, call_planning_agent_with_feedback
from agents.PDAS import call_single_task_agent
from agents.output_generator import call_output_generator_agent
from agents.PDAS import call_evaluation_agent
from agents.score import call_score_agent
from agents.auto_user import call_auto_user_agent, load_random_user_profile


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _tok(result) -> dict:
    """
    Devolve apenas o delta de tokens desta chamada.
    O LangGraph soma automaticamente via o reducer operator.add definido em state.py.
    NUNCA acumular manualmente — causaria dupla contagem.
    """
    return {
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
        "total_tiktoken_input": result.tiktoken_input,
        "total_tiktoken_output": result.tiktoken_output,
    }


def _log_entry(agent: str, inp: Any, out: str) -> dict:
    """Cria uma entrada para agent_io_log."""
    return {"agent": agent, "input": inp, "output": out}


# ---------------------------------------------------------------------------
# Nó: saudação inicial
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
# Nó: input do utilizador (humano ou simulado)
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
# Nó: validação da mensagem do utilizador
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
# Nó: detecção de intenção
# ---------------------------------------------------------------------------
def node_detect_intent(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    context = state.get("context", [])
    result  = call_intent_agent(context, run_cfg, run_cfg.use_full_context)
    parsed  = safe_json_loads(result.content) or {}
    intent  = parsed.get("intent", "UNDEFINED")

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] intent_agent → {result.content}", "DEBUG")

    log_mgr.record("intent_agent", context[-1]["message"], result.content)

    attempts = state.get("intent_attempts", 0)
    if intent == "UNDEFINED":
        attempts += 1
    else:
        attempts = 0

    return {
        **_tok(result),
        "total_tiktoken_input":     result.tiktoken_input,
        "total_tiktoken_output":    result.tiktoken_output,
        "intent":                   intent,
        "intent_attempts":          attempts,
        "agent_io_log":             [_log_entry("intent_agent", context[-1]["message"], result.content)],
    }


# ---------------------------------------------------------------------------
# Nó: resposta de intenção não reconhecida (via InterfaceAgent)
# ---------------------------------------------------------------------------
def node_intent_unclear(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    context     = state.get("context", [])
    instruction = "Adapte a seguinte mensagem, mantendo o estilo do utilizador: Desculpe, não entendi. Pode reformular o que precisa?"
    result      = call_interface_agent(context, instruction, run_cfg, run_cfg.use_full_context)
    msg         = result.content

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] interface_agent (intent_unclear) → {msg}", "DEBUG")

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
# Nó: carregamento do formulário
# ---------------------------------------------------------------------------
def node_load_form(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    intent    = state.get("intent", "1")
    form_name = f"formulario_{intent}"
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
# Nó: planning (sem feedback)
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
# Nó: planning com feedback
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
# Nó: execução de uma etapa do plano (SingleTaskAgent + InterfaceAgent)
# ---------------------------------------------------------------------------
def node_execute_stage(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    plan  = state.get("plan", [])
    idx   = state.get("current_stage_index", 0)
    ctx   = state.get("context", [])
    form  = state.get("form", {})

    if idx >= len(plan):
        return {}

    stage      = plan[idx]
    task_input = json.dumps(state.get("single_task_output") or form, ensure_ascii=False)

    # SingleTaskAgent
    st_result = call_single_task_agent(
        task_action          = stage["action"],
        task_input           = task_input,
        task_expected_output = stage["output"],
        run_cfg              = run_cfg,
    )
    if run_cfg.debug_mode:
        color_print(f"[DEBUG] single_task_agent (stage {idx+1}) → {st_result.content[:80]}", "DEBUG")
    log_mgr.record("single_task_agent", stage, st_result.content)

    # InterfaceAgent — adapta ao estilo do utilizador
    instruction = f"Adapte a seguinte mensagem, mantendo o estilo de linguagem do utilizador: {st_result.content}"
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
        "single_task_output":       st_result.content,
        **_tok_multi(st_result, if_result),
        "total_tiktoken_input":     st_result.tiktoken_input + if_result.tiktoken_input,
        "total_tiktoken_output":    st_result.tiktoken_output + if_result.tiktoken_output,
        "total_input_tokens":       st_result.input_tokens  + if_result.input_tokens,
        "total_output_tokens":      st_result.output_tokens + if_result.output_tokens,
        "agent_io_log": [
            _log_entry("single_task_agent", stage, st_result.content),
            _log_entry("interface_agent", instruction, msg),
        ],
    }


# ---------------------------------------------------------------------------
# Nó: guardar resposta do utilizador para a etapa atual
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
    }


# ---------------------------------------------------------------------------
# Nó: output generator
# ---------------------------------------------------------------------------
def _merge_answers_into_form(form: dict, flat_answers: dict, enriched_answers: dict) -> dict:
    """
    Tenta preencher o formulário original com respostas vindas de um dict plano.

    Estratégias por ordem:
    1. Mapear por campo_id exacto (campo1, campo2...)
    2. Mapear por semelhança de label (case-insensitive)
    3. Usar o contexto da conversa para extrair valor para cada campo
    """
    import copy

    filled       = copy.deepcopy(form)
    label_map: dict[str, dict] = {}
    for secao in filled.get("secoes", []):
        for campo in secao.get("campos", []):
            label_map[campo.get("label", "").lower()] = campo
            label_map[campo.get("id",    "").lower()] = campo

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


def node_generate_output(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    form    = state.get("form", {})
    answers = state.get("form_answers", {})
    context = state.get("context", [])

    context_str = "\n".join(
        f"{m['who']}: {m['message']}" for m in context
        if m["who"] in ("CITIZEN", "CHATBOT")
    )
    enriched_answers = {
        "stage_answers": answers,
        "conversation":  context_str,
    }

    result = call_output_generator_agent(form, enriched_answers, run_cfg)
    parsed = safe_json_loads(result.content)

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] output_generator_agent → {result.content[:80]}", "DEBUG")

    log_mgr.record("output_generator_agent", enriched_answers, result.content)

    if parsed and "secoes" in parsed:
        filled = parsed
        errors = []
    elif parsed and isinstance(parsed, dict):
        filled = _merge_answers_into_form(form, parsed, enriched_answers)
        errors = []
    else:
        filled = {}
        errors = ["output_generator_agent devolveu JSON inválido"]

    return {
        **_tok(result),
        "total_tiktoken_input": result.tiktoken_input,
        "total_tiktoken_output": result.tiktoken_output,
        "filled_form":  filled,
        "errors":       errors,
        "agent_io_log": [_log_entry("output_generator_agent", "answers", result.content)],
    }


# ---------------------------------------------------------------------------
# Nó: cálculo de métricas (hit_rate)
# ---------------------------------------------------------------------------
def node_calculate_metrics(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    filled       = state.get("filled_form") or state.get("form_answers", {})
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

    return {"hit_rate": hr, "report": rep, "advanced_metrics": adv}


# ---------------------------------------------------------------------------
# Nó: evaluation agent
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
# Nó: score agent
# ---------------------------------------------------------------------------
def node_score(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    filled = state.get("filled_form") or state.get("form_answers", {})
    result = call_score_agent(filled, run_cfg)

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] score_agent → {result.content}", "DEBUG")

    log_mgr.record("score_agent", "filled_form", result.content)

    return {
        **_tok(result),
        "total_tiktoken_input": result.tiktoken_input,
        "total_tiktoken_output": result.tiktoken_output,
        "score":        result.content.strip(),
        "agent_io_log": [_log_entry("score_agent", "filled_form", result.content)],
    }


# ---------------------------------------------------------------------------
# Funções de routing (usadas nas conditional_edges)
# ---------------------------------------------------------------------------
def route_after_validation(state: FormFillingState) -> str:
    """Após validação: continua ou pede nova resposta."""
    attempts = state.get("validation_attempts", 0)
    max_att  = 3  # RunConfig.max_validation_attempts
    if attempts >= max_att:
        return "force_next"
    logs = state.get("agent_io_log", [])
    for entry in reversed(logs):
        if entry.get("agent") == "validation_agent":
            parsed = safe_json_loads(entry["output"]) or {}
            if parsed.get("status") == "VALIDATED":
                return "validated"
            return "not_validated"
    return "validated"  # fallback


def route_after_intent(state: FormFillingState) -> str:
    """Após detecção de intenção: formulário encontrado ou intenção indefinida."""
    intent   = state.get("intent", "UNDEFINED")
    attempts = state.get("intent_attempts", 0)
    max_att  = 5
    if attempts >= max_att:
        return "max_attempts"
    if intent == "UNDEFINED":
        return "undefined"
    return "found"


def make_route_stage_or_done(run_cfg: RunConfig):
    """
    Fábrica de routing para verificar se ainda há etapas do plano a executar.

    Devolve um callable compatível com LangGraph que usa run_cfg.max_plan_stages
    em vez de um valor hardcoded. Usar como:

        g.add_conditional_edges(
            "store_answer",
            make_route_stage_or_done(run_cfg),
            {"next_stage": "execute_stage", "done": "generate_output"},
        )
    """
    def _route(state: FormFillingState) -> str:
        idx  = state.get("current_stage_index", 0)
        plan = state.get("plan", [])
        if idx >= len(plan) or idx >= run_cfg.max_plan_stages:
            return "done"
        return "next_stage"
    return _route

def route_stage_or_done(state: FormFillingState) -> str:
    """DEPRECADO: usa max_plan_stages=20 hardcoded. Ver make_route_stage_or_done."""
    idx  = state.get("current_stage_index", 0)
    plan = state.get("plan", [])
    if idx >= len(plan) or idx >= 20:
        return "done"
    return "next_stage"