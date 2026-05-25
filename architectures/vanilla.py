"""
architectures/vanilla.py — Vanilla LLM

Um único agente monolítico gere toda a conversa.
A cada turno recebe o formulário + respostas acumuladas + última mensagem do utilizador
e decide a próxima pergunta (ou termina).

Fluxo:
  greet → get_input → detect_intent
    └─ found → vanilla_turn ←────────────────┐
                   ├─ in_progress → get_input ┘
                   └─ complete    → metrics → score → END
"""

from __future__ import annotations
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from string import Template
import json
import time

from langgraph.graph import StateGraph, END

from config import RunConfig
from graphs.state import FormFillingState, initial_state
from graphs.shared_nodes import (
    node_greet, node_get_user_input, node_detect_intent,
    node_intent_unclear, node_load_form,
    node_generate_output, node_calculate_metrics, node_score,
    route_after_intent,
)
from agents.base import call_llm
from utils.logging import RunLog, LogFileManager, color_print
from utils.forms import write_form, safe_json_loads


# ---------------------------------------------------------------------------
# Prompt do VanillaAgent (inline — não tem ficheiro de prompt próprio)
# ---------------------------------------------------------------------------
_VANILLA_PROMPT = Template("""
Você é um assistente útil encarregado de preencher um formulário com base na entrada do utilizador.

REGRAS CRÍTICAS:
- "current_answers" contém TODAS as respostas já recolhidas. NUNCA as apague ou substitua.
- "new_answer" deve conter APENAS a resposta nova desta mensagem (se houver).
  O sistema faz o merge automaticamente — não repita campos já respondidos.
- Identifique o próximo campo obrigatório ainda sem resposta e faça uma pergunta clara.
- Inclua as opções disponíveis quando o campo for "radio" ou "checkbox".
- Quando TODOS os campos obrigatórios tiverem resposta em "current_answers" + "new_answer", defina status="complete".

O formulário é:
$form_json

Respostas já recolhidas (NÃO altere):
$answers_json

Última mensagem do utilizador (processe esta resposta):
$last_message

Responda APENAS com este JSON, sem explicações:
{
  "new_answer": {"campo_id": "resposta desta mensagem"},
  "status": "in_progress" ou "complete",
  "next_message": "próxima pergunta ou mensagem de conclusão"
}
""")


# ---------------------------------------------------------------------------
# Nó: turno do VanillaAgent
# ---------------------------------------------------------------------------
def node_vanilla_turn(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    form    = state.get("form", {})
    answers = state.get("form_answers", {})
    context = state.get("context", [])

    last_user = next(
        (m["message"] for m in reversed(context) if m["who"] == "CITIZEN"),
        "Nenhuma (início do preenchimento)",
    )

    prompt = _VANILLA_PROMPT.substitute(
        form_json    = json.dumps(form,    indent=2, ensure_ascii=False),
        answers_json = json.dumps(answers, indent=2, ensure_ascii=False),
        last_message = last_user,
    )

    model  = run_cfg.model_for_agent("vanilla_agent")
    result = call_llm(prompt, model=model)
    parsed = safe_json_loads(result.content) or {}

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] vanilla_agent → {result.content[:100]}", "DEBUG")

    log_mgr.record("vanilla_agent", last_user, result.content)

    new_answer = parsed.get("new_answer") or {}
    merged     = {**answers, **new_answer}
    status     = parsed.get("status", "in_progress")
    msg        = parsed.get("next_message", "Erro ao gerar mensagem.")

    # Segurança anti-loop: forçar conclusão após demasiadas iterações
    iteration = state.get("iteration_count", 0) + 1
    if iteration >= run_cfg.max_plan_stages * 2 and status != "complete":
        color_print("[WARN] Limite de iterações atingido — forçando conclusão.", "ERROR")
        status = "complete"

    color_print(msg, "CHATBOT")
    new_ctx = context + [{"who": "CHATBOT", "message": msg}]
    log_mgr.write("chat", f"CHATBOT: {msg}\n")

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] respostas acumuladas: {list(merged.keys())}", "DEBUG")

    return {
        "form_answers":        merged,
        "context":             new_ctx,
        "vanilla_status":      status,
        "iteration_count":     iteration,
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def route_vanilla(state: FormFillingState) -> str:
    status = state.get("vanilla_status", "in_progress")
    return "complete" if status == "complete" else "in_progress"


# ---------------------------------------------------------------------------
# Construção do grafo
# ---------------------------------------------------------------------------
def build_graph(run_cfg: RunConfig, log_mgr: LogFileManager) -> StateGraph:
    def _node(fn):
        return partial(fn, run_cfg=run_cfg, log_mgr=log_mgr)

    g = StateGraph(FormFillingState)

    g.add_node("greet",           _node(node_greet))
    g.add_node("get_input",       _node(node_get_user_input))
    g.add_node("detect_intent",   _node(node_detect_intent))
    g.add_node("intent_unclear",  _node(node_intent_unclear))
    g.add_node("load_form",       _node(node_load_form))
    g.add_node("vanilla_turn",    _node(node_vanilla_turn))
    g.add_node("get_input_v",     _node(node_get_user_input))
    g.add_node("generate_output", _node(node_generate_output))
    g.add_node("metrics",         _node(node_calculate_metrics))
    g.add_node("score",           _node(node_score))

    g.set_entry_point("greet")

    g.add_edge("greet",           "get_input")
    g.add_edge("get_input",       "detect_intent")
    g.add_edge("intent_unclear",  "get_input")
    g.add_edge("load_form",       "vanilla_turn")
    g.add_edge("get_input_v",     "vanilla_turn")
    g.add_edge("generate_output", "metrics")
    g.add_edge("metrics",         "score")
    g.add_edge("score",           END)

    g.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {"found": "load_form", "undefined": "intent_unclear", "max_attempts": END},
    )

    g.add_conditional_edges(
        "vanilla_turn",
        route_vanilla,
        {"in_progress": "get_input_v", "complete": "generate_output"},
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
    filled    = final_state.get("filled_form") or {}

    run_log = RunLog()
    run_log.metricas.total_input_tokens            = final_state.get("total_input_tokens", 0)
    run_log.metricas.total_output_tokens           = final_state.get("total_output_tokens", 0)
    run_log.metricas.total_tiktoken_input          = final_state.get("total_tiktoken_input", 0)
    run_log.metricas.total_tiktoken_output         = final_state.get("total_tiktoken_output", 0)
    run_log.metricas.tempo_total_execucao_segundos = elapsed
    run_log.metricas.numero_iteracoes              = final_state.get("iteration_count", 0)
    run_log.metricas.hit_rate                      = final_state.get("hit_rate", 0.0)
    run_log.advanced_metrics                       = final_state.get("advanced_metrics", {})
    run_log.respostas_formulario                   = filled
    run_log.output                                 = filled
    run_log.contexto_final                         = final_state.get("context", [])
    run_log.report                                 = final_state.get("report", {})
    run_log.erros                                  = final_state.get("errors", [])
    run_log.architecture                           = run_cfg.architecture.value
    run_log.auto_user_enabled                      = run_cfg.auto_user
    run_log.auto_user_model                        = run_cfg.auto_user_model if run_cfg.auto_user else None
    run_log.models_per_agent = {
        "vanilla_agent":                             run_cfg.model_for_agent("vanilla_agent"),
    }

    write_form(form_name, filled, run_dir)
    log_mgr.save_metrics(run_log)
    log_mgr.save_json_log(run_log, run_dir)

    color_print(f"Execução concluída!", "INFO")
    color_print(f"Tempo: {elapsed}s  |  Iterações: {run_log.metricas.numero_iteracoes}  |  Hit rate: {run_log.metricas.hit_rate:.2%}  |  Score: {final_state.get('score', 'n/a')}", "INFO")
    color_print(f"Eficiência de perguntas: {run_log.advanced_metrics.get('question_efficiency')}  |  Conformidade condicional: {run_log.advanced_metrics.get('conditional_compliance')}", "INFO")    
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