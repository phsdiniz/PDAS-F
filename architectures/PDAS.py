"""
architectures/PDAS.py — PDAS (sem feedback)

Fluxo:
  greet → get_input → detect_intent
    ├─ UNDEFINED → intent_unclear → get_input (loop)
    ├─ max_attempts → END
    └─ found → load_form → plan → execute_stage → get_input_stage → validate
                   ↑                    ↑               └─ validated   → store_answer
                   └────────────────────┘  not_validated → get_input_stage (loop)
                                        └─ done → generate_output → metrics → score → END
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
    node_greet, node_get_user_input, node_validate, node_detect_intent,
    node_intent_unclear, node_load_form, node_plan, node_execute_stage,
    node_store_answer, node_generate_output, node_calculate_metrics, node_score,
    route_after_intent, route_after_validation, make_route_stage_or_done,
)
from utils.logging import RunLog, LogFileManager, color_print
from utils.forms import write_form


# ---------------------------------------------------------------------------
# Construção do grafo
# ---------------------------------------------------------------------------
def build_graph(run_cfg: RunConfig, log_mgr: LogFileManager) -> StateGraph:
    """
    Constrói e compila o grafo LangGraph para a arquitectura PDAS.
    Cada nó é um partial que já tem run_cfg e log_mgr injetados.
    """
    def _node(fn):
        return partial(fn, run_cfg=run_cfg, log_mgr=log_mgr)

    g = StateGraph(FormFillingState)

    # --- Adicionar nós ---
    g.add_node("greet",           _node(node_greet))
    g.add_node("get_input",       _node(node_get_user_input))
    g.add_node("detect_intent",   _node(node_detect_intent))
    g.add_node("intent_unclear",  _node(node_intent_unclear))
    g.add_node("load_form",       _node(node_load_form))
    g.add_node("plan",            _node(node_plan))
    g.add_node("execute_stage",   _node(node_execute_stage))
    g.add_node("get_input_stage", _node(node_get_user_input))
    g.add_node("validate",        _node(node_validate))
    g.add_node("store_answer",    _node(node_store_answer))
    g.add_node("generate_output", _node(node_generate_output))
    g.add_node("metrics",         _node(node_calculate_metrics))
    g.add_node("score",           _node(node_score))

    # --- Entry point ---
    g.set_entry_point("greet")

    # --- Arestas fixas ---
    g.add_edge("greet",           "get_input")
    g.add_edge("get_input",       "detect_intent")
    g.add_edge("intent_unclear",  "get_input")
    g.add_edge("load_form",       "plan")
    g.add_edge("plan",            "execute_stage")
    g.add_edge("execute_stage",   "get_input_stage")
    g.add_edge("get_input_stage", "validate")
    g.add_edge("generate_output", "metrics")
    g.add_edge("metrics",         "score")
    g.add_edge("score",           END)

    # --- Arestas condicionais ---
    g.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "found":        "load_form",
            "undefined":    "intent_unclear",
            "max_attempts": END,
        },
    )

    # Routing após validação — validated/force_next avançam, not_validated repete
    g.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "validated":    "store_answer",
            "not_validated": "get_input_stage",
            "force_next":   "store_answer",
        },
    )

    # Usar make_route_stage_or_done(run_cfg) em vez de route_stage_or_done
    # para respeitar run_cfg.max_plan_stages em vez de 20 hardcoded.
    g.add_conditional_edges(
        "store_answer",
        make_route_stage_or_done(run_cfg),
        {
            "next_stage": "execute_stage",
            "done":       "generate_output",
        },
    )

    return g.compile()


# ---------------------------------------------------------------------------
# Runner da arquitectura
# ---------------------------------------------------------------------------call_planning_agent
def run(stack: ExitStack, run_dir: Path, run_cfg: RunConfig, langfuse_callback=None) -> RunLog:
    """
    Executa uma simulação completa no modo PDAS.

    Returns:
        RunLog com métricas e resultados da run.
    """

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
    run_log.intent_detectado                       = final_state.get("intent")
    run_log.plano_inicial                          = final_state.get("plan")
    run_log.respostas_formulario                   = final_state.get("form_answers", {})
    run_log.contexto_final                         = final_state.get("context", [])
    run_log.output                                 = filled
    run_log.report                                 = final_state.get("report", {})
    run_log.erros                                  = final_state.get("errors", [])
    run_log.user_profile                           = final_state.get("user_profile")
    run_log.architecture                           = run_cfg.architecture.value
    run_log.auto_user_enabled                      = run_cfg.auto_user
    run_log.auto_user_model                        = run_cfg.auto_user_model if run_cfg.auto_user else None
    run_log.models_per_agent = {
        "intent_agent":                              run_cfg.model_for_agent("intent_agent"),
        "interface_agent":                           run_cfg.model_for_agent("interface_agent"),
        "validation_agent":                          run_cfg.model_for_agent("validation_agent"),
        "output_generator":                          run_cfg.model_for_agent("output_generator"),
        "score_agent":                               run_cfg.model_for_agent("score_agent"),
        "planning_agent":                            run_cfg.model_for_agent("planning_agent"),
        "single_task_agent":                         run_cfg.model_for_agent("single_task_agent"),
        "evaluation_agent":                          run_cfg.model_for_agent("evaluation_agent"),
    }

    log_mgr.save_metrics(run_log)
    log_mgr.save_json_log(run_log, run_dir)

    color_print(f"Execução concluída!", "INFO")
    color_print(f"Tempo: {elapsed}s  |  Iterações: {run_log.metricas.numero_iteracoes}  |  Hit rate: {run_log.metricas.hit_rate:.2%}  |  Score: {final_state.get('score', 'n/a')}", "INFO")
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