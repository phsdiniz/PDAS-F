"""
architectures/PDAS_F.py — PDAS with inter-session feedback

Same as PDAS (including the per-turn dual validation and within-session
replanning — see architectures/PDAS.py), plus:
  1. Before planning, loads the previous run's feedback and plan (from disk).
  2. Uses PlanningAgentWithFeedback instead of PlanningAgent.
  3. After evaluating, persists the feedback and plan in RESULTS_DIR for the next run.

Additional flow vs. PDAS:
  load_form → load_feedback → plan_with_feedback → (rest same as PDAS)
                                                  → evaluate → save_feedback → END
"""

from __future__ import annotations
from contextlib import ExitStack
from functools import partial
from pathlib import Path
import json
import time

from langgraph.graph import StateGraph, END

from config import RunConfig, RESULTS_DIR
from graphs.state import FormFillingState, initial_state
from graphs.shared_nodes import (
    node_greet, node_get_user_input, node_validate, node_identify_form,
    node_form_unclear, node_load_form, node_plan_with_feedback,
    node_execute_single_task, node_validate_output, node_replan_stage,
    node_escalate_output_validation, node_send_stage_message,
    node_escalate_validation,
    node_store_answer,
    node_calculate_metrics, node_evaluate,
    make_route_after_form_identification, make_route_after_validation,
    make_route_after_output_validation, make_route_stage_or_done,
)
from utils.logging import RunLog, LogFileManager, color_print
from utils.forms import write_form


# ---------------------------------------------------------------------------
# Nodes specific to this architecture
# ---------------------------------------------------------------------------
def node_load_feedback(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Loads the previous feedback and plan from disk (if they exist)."""
    feedback_path = RESULTS_DIR / "evaluation_agent_output.txt"
    plan_path     = RESULTS_DIR / "previous_plan.txt"

    feedback      = feedback_path.read_text(encoding="utf-8") if feedback_path.exists() else ""
    previous_plan = plan_path.read_text(encoding="utf-8")     if plan_path.exists()     else ""

    if run_cfg.debug_mode and feedback:
        color_print(f"[DEBUG] Feedback anterior carregado ({len(feedback)} chars)", "DEBUG")

    return {"feedback": feedback, "previous_plan": previous_plan}


def node_save_feedback(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Persists this run's feedback and plan for the next one."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feedback = state.get("feedback", "")
    plan_str = state.get("plan_str", "")

    if feedback:
        (RESULTS_DIR / "evaluation_agent_output.txt").write_text(feedback, encoding="utf-8")
    if plan_str:
        (RESULTS_DIR / "previous_plan.txt").write_text(plan_str, encoding="utf-8")

    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(run_cfg: RunConfig, log_mgr: LogFileManager) -> StateGraph:
    def _node(fn):
        return partial(fn, run_cfg=run_cfg, log_mgr=log_mgr)

    g = StateGraph(FormFillingState)

    g.add_node("greet",               _node(node_greet))
    g.add_node("get_input",           _node(node_get_user_input))
    g.add_node("identify_form",       _node(node_identify_form))
    g.add_node("form_unclear",        _node(node_form_unclear))
    g.add_node("load_form",           _node(node_load_form))
    g.add_node("load_feedback",       _node(node_load_feedback))
    g.add_node("plan",                _node(node_plan_with_feedback))
    g.add_node("execute_single_task", _node(node_execute_single_task))
    g.add_node("validate_output",     _node(node_validate_output))
    g.add_node("replan_stage",        _node(node_replan_stage))
    g.add_node("escalate_output",     _node(node_escalate_output_validation))
    g.add_node("send_stage_message",  _node(node_send_stage_message))
    g.add_node("get_input_stage",     _node(node_get_user_input))
    g.add_node("validate",            _node(node_validate))
    g.add_node("escalate_validation", _node(node_escalate_validation))
    g.add_node("store_answer",        _node(node_store_answer))
    g.add_node("metrics",             _node(node_calculate_metrics))
    g.add_node("evaluate",            _node(node_evaluate))
    g.add_node("save_feedback",       _node(node_save_feedback))

    g.set_entry_point("greet")

    # Fixed edges
    g.add_edge("greet",               "get_input")
    g.add_edge("get_input",           "identify_form")
    g.add_edge("form_unclear",        "get_input")
    g.add_edge("load_form",           "load_feedback")
    g.add_edge("load_feedback",       "plan")
    g.add_edge("plan",                "execute_single_task")
    g.add_edge("execute_single_task", "validate_output")
    g.add_edge("replan_stage",        "execute_single_task")
    g.add_edge("escalate_output",     "send_stage_message")
    g.add_edge("send_stage_message",  "get_input_stage")
    g.add_edge("get_input_stage",     "validate")
    g.add_edge("escalate_validation", "store_answer")
    g.add_edge("metrics",             "evaluate")
    g.add_edge("evaluate",            "save_feedback")
    g.add_edge("save_feedback",       END)

    # Conditional edges
    g.add_conditional_edges(
        "identify_form",
        make_route_after_form_identification(run_cfg),
        {
            "found":        "load_form",
            "undefined":    "form_unclear",
            "max_attempts": END,
        },
    )

    # Task Agent output validation — see architectures/PDAS.py
    g.add_conditional_edges(
        "validate_output",
        make_route_after_output_validation(run_cfg),
        {
            "validated":     "send_stage_message",
            "not_validated": "replan_stage",
            "escalated":      "escalate_output",
        },
    )

    # Routing after validating the user's message
    g.add_conditional_edges(
        "validate",
        make_route_after_validation(run_cfg),
        {
            "validated":      "store_answer",
            "not_validated":  "get_input_stage",
            "escalated":      "escalate_validation",
        },
    )

    g.add_conditional_edges(
        "store_answer",
        make_route_stage_or_done(run_cfg),
        {
            "next_stage": "execute_single_task",
            "done":       "metrics",
        },
    )

    return g.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(stack: ExitStack, run_dir: Path, run_cfg: RunConfig, langfuse_callback=None) -> RunLog:
    log_mgr    = LogFileManager(run_dir, stack, architecture=run_cfg.architecture.value)
    graph      = build_graph(run_cfg, log_mgr)
    state      = initial_state()
    start_time = time.time()

    invoke_config = {"callbacks": [langfuse_callback]} if langfuse_callback else {}
    final_state: FormFillingState = graph.invoke(state, config=invoke_config)
    elapsed = round(time.time() - start_time, 2)

    form_name = final_state.get("form_name", "formulario")
    filled    = final_state.get("filled_form") or final_state.get("form_answers", {})
    write_form(form_name, filled, run_dir)

    run_log = RunLog()
    run_log.metricas.total_input_tokens            = final_state.get("total_input_tokens", 0)
    run_log.metricas.total_output_tokens           = final_state.get("total_output_tokens", 0)
    run_log.metricas.total_tiktoken_input          = final_state.get("total_tiktoken_input", 0)
    run_log.metricas.total_tiktoken_output         = final_state.get("total_tiktoken_output", 0)
    run_log.metricas.tempo_total_execucao_segundos = elapsed
    run_log.metricas.numero_iteracoes              = final_state.get("iteration_count", 0)
    run_log.metricas.hit_rate                      = final_state.get("hit_rate", 0.0)
    run_log.advanced_metrics                       = final_state.get("advanced_metrics", {})
    run_log.form_id_detectado    = final_state.get("form_id")
    run_log.plano_inicial        = final_state.get("plan")
    run_log.respostas_formulario = final_state.get("form_answers", {})
    run_log.contexto_final       = final_state.get("context", [])
    run_log.output               = filled
    run_log.report               = final_state.get("report", {})
    run_log.erros                = final_state.get("errors", [])
    run_log.user_profile         = final_state.get("user_profile")
    run_log.architecture                           = run_cfg.architecture.value
    run_log.auto_user_enabled                      = run_cfg.auto_user
    run_log.auto_user_model                        = run_cfg.auto_user_model if run_cfg.auto_user else None
    run_log.models_per_agent = {
        "interface_agent":                           run_cfg.model_for_agent("interface_agent"),
        "validation_agent":                          run_cfg.model_for_agent("validation_agent"),
        "planning_agent_with_feedback":              run_cfg.model_for_agent("planning_agent"),
        "single_task_agent":                         run_cfg.model_for_agent("single_task_agent"),
        "evaluation_agent":                          run_cfg.model_for_agent("evaluation_agent"),
    }

    log_mgr.save_metrics(run_log)
    log_mgr.save_json_log(run_log, run_dir)

    color_print(f"Execução concluída!", "INFO")
    color_print(f"Tempo: {elapsed}s  |  Iterações: {run_log.metricas.numero_iteracoes}  |  Hit rate: {run_log.metricas.hit_rate:.2%}", "INFO")
    color_print(f"Eficiência de perguntas: {run_log.advanced_metrics.get('question_efficiency', 'N/A'):.2f}  |  Conformidade condicional: {run_log.advanced_metrics.get('conditional_compliance', 'N/A'):.2%}", "INFO")    
    color_print("Formulário preenchido:", "INFO")
    print(json.dumps(filled, indent=2, ensure_ascii=False))
    if run_log.report:
        color_print("\nRelatório de campos:", "INFO")
        for campo, status in run_log.report.items():
            print(f"  {campo}: {status}")
    if run_cfg.debug_mode and run_log.erros:
        color_print(f"Erros: {run_log.erros}", "DEBUG")

    print(run_log.advanced_metrics)

    print(run_log.metricas)

    return run_log
