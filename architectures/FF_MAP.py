"""
architectures/FF_MAP.py — FF_MAP

Rede fixa de agentes especializados:
  generate_questions → para cada campo:
      ask_question → get_input → parse_answer
          ├─ information_extraction → store_extracted → next_field
          ├─ follow_up_question     → get_input → parse_answer (loop)
          └─ repeat_question        → get_input → parse_answer (loop)
  → fill_form → generate_json → metrics → score → END
"""

from __future__ import annotations
from contextlib import ExitStack
from functools import partial
from pathlib import Path
import json
import time

from langgraph.graph import StateGraph, END

from config import RunConfig
from graphs.state import FormFillingState, initial_state
from graphs.shared_nodes import (
    node_greet, node_get_user_input, node_detect_intent,
    node_intent_unclear, node_load_form,
    node_calculate_metrics, node_score,
    route_after_intent,
)
from agents.FF_MAP import (
    call_question_generation_agent,
    call_answer_parsing_agent,
    call_information_extraction_agent,
    call_follow_up_question_agent,
    call_repeat_question_agent,
    call_form_filling_agent,
    call_json_generation_agent,
)
from utils.logging import RunLog, LogFileManager, color_print
from utils.forms import write_form, safe_json_loads


# ---------------------------------------------------------------------------
# Nós específicos desta arquitectura
# ---------------------------------------------------------------------------

def node_generate_questions(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Gera lista de perguntas e campos obrigatórios a partir do formulário."""
    form   = state.get("form", {})
    result = call_question_generation_agent(form, run_cfg)
    log_mgr.record("question_generation_agent", "form", result.content, RunLog())

    # Extrai perguntas e ids dos campos obrigatórios diretamente do formulário
    questions, fields = [], []
    for secao in form.get("secoes", []):
        for campo in secao.get("campos", []):
            if campo.get("required"):
                q = f"{campo['label']} (Obrigatório)"
                if campo.get("options"):
                    q += f" Opções: {', '.join(campo['options'])}"
                questions.append(q + "?")
                fields.append(campo["id"])

    return {
        "questions":           questions,
        "fields":              fields,
        "current_field_index": 0,
        "extracted_info":      [],
        "current_dialogue":    "",
        "parse_action":        "",
        "total_input_tokens":  state.get("total_input_tokens",  0) + result.input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + result.output_tokens,
    }


def node_ask_question(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Apresenta a pergunta atual ao utilizador."""
    questions = state.get("questions", [])
    idx       = state.get("current_field_index", 0)
    ctx       = state.get("context", [])

    if idx >= len(questions):
        return {}

    question = questions[idx]
    color_print(question, "CHATBOT")
    new_ctx = ctx + [{"who": "CHATBOT", "message": question}]
    log_mgr.write("chat", f"CHATBOT: {question}\n")

    return {"context": new_ctx}


def node_parse_answer(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Decide a próxima ação: extraction, follow_up ou repeat."""
    ctx    = state.get("context", [])
    fields = state.get("fields", [])
    idx    = state.get("current_field_index", 0)
    form   = state.get("form", {})

    bot_msgs  = [m for m in ctx if m["who"] == "CHATBOT"]
    user_msgs = [m for m in ctx if m["who"] == "CITIZEN"]
    question  = bot_msgs[-1]["message"]  if bot_msgs  else ""
    answer    = user_msgs[-1]["message"] if user_msgs else ""
    dialogue  = f"Pergunta: {question}\nResposta: {answer}"
    field_id  = fields[idx] if idx < len(fields) else ""

    #current_fields = fields[idx] if isinstance(fields[idx], list) else [fields[idx]]
    result = call_answer_parsing_agent(dialogue, field_id, form, run_cfg) # passar current_fields para call_answer_parsing_agent definido na linha anterior?
    parsed = safe_json_loads(result.content) or {}
    action = parsed.get("next action", "information_extraction")

    if run_cfg.debug_mode:
        color_print(f"[DEBUG] answer_parsing → {action}", "DEBUG")

    log_mgr.record("answer_parsing_agent", dialogue, result.content, None)

    return {
        "current_dialogue":    dialogue,
        "parse_action":        action,
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def node_extract_information(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Extrai informação relevante do diálogo e avança para o próximo campo."""
    dialogue = state.get("current_dialogue", "")
    result   = call_information_extraction_agent(dialogue, run_cfg)
    log_mgr.record("information_extraction_agent", dialogue, result.content, None)

    extracted = list(state.get("extracted_info", [])) + [result.content]
    idx       = state.get("current_field_index", 0)

    return {
        "extracted_info":      extracted,
        "current_field_index": idx + 1,
        "iteration_count":     state.get("iteration_count", 0) + 1,
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def node_follow_up_question(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Gera e apresenta uma pergunta de seguimento."""
    ctx      = state.get("context", [])
    fields   = state.get("fields", [])
    idx      = state.get("current_field_index", 0)
    form     = state.get("form", {})
    dialogue = state.get("current_dialogue", "")
    field_id = fields[idx] if idx < len(fields) else ""

    result   = call_follow_up_question_agent(dialogue, field_id, form, run_cfg)
    parsed   = safe_json_loads(result.content) or {}
    question = parsed.get("question", "Pode dar mais detalhes?")

    color_print(question, "CHATBOT")
    new_ctx = ctx + [{"who": "CHATBOT", "message": question}]
    log_mgr.write("chat", f"CHATBOT: {question}\n")
    log_mgr.record("follow_up_question_agent", dialogue, result.content, None)

    return {
        "context":             new_ctx,
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def node_repeat_question(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Reformula a pergunta atual quando a resposta foi inválida."""
    ctx      = state.get("context", [])
    fields   = state.get("fields", [])
    idx      = state.get("current_field_index", 0)
    form     = state.get("form", {})
    dialogue = state.get("current_dialogue", "")
    field_id = fields[idx] if idx < len(fields) else ""

    result   = call_repeat_question_agent(dialogue, field_id, form, run_cfg)
    parsed   = safe_json_loads(result.content) or {}
    question = parsed.get(
        "new question",
        state.get("questions", [""])[idx] if idx < len(state.get("questions", [])) else "Pode repetir?",
    )

    color_print(question, "CHATBOT")
    new_ctx = ctx + [{"who": "CHATBOT", "message": question}]
    log_mgr.write("chat", f"CHATBOT: {question}\n")
    log_mgr.record("repeat_question_agent", dialogue, result.content, None)

    return {
        "context":             new_ctx,
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def node_fill_form(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Mapeia informação extraída para campos do formulário."""
    form      = state.get("form", {})
    extracted = state.get("extracted_info", [])
    fields    = state.get("fields", [])

    result = call_form_filling_agent(form, extracted, fields, run_cfg)
    log_mgr.record("form_filling_agent", extracted, result.content, None)

    return {
        "filled_form_text":    result.content,
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def node_generate_json(
    state: FormFillingState,
    run_cfg: RunConfig,
    log_mgr: LogFileManager,
) -> dict:
    """Gera o formulário final em JSON válido."""
    form            = state.get("form", {})
    filled_form_txt = state.get("filled_form_text", "")

    result = call_json_generation_agent(form, filled_form_txt, run_cfg)
    parsed = safe_json_loads(result.content) or {}
    log_mgr.record("json_generation_agent", filled_form_txt, result.content, None)

    return {
        "filled_form":         parsed,
        "form_answers":        parsed,
        "total_input_tokens":  result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_parse_action(state: FormFillingState) -> str:
    action = state.get("parse_action", "information_extraction")
    if action == "follow_up_question":
        return "follow_up"
    if action == "repeat_question":
        return "repeat"
    return "extract"


def route_next_field_or_done(state: FormFillingState) -> str:
    idx    = state.get("current_field_index", 0)
    fields = state.get("fields", [])
    return "done" if idx >= len(fields) else "next_field"


# ---------------------------------------------------------------------------
# Construção do grafo
# ---------------------------------------------------------------------------
def build_graph(run_cfg: RunConfig, log_mgr: LogFileManager) -> StateGraph:
    def _node(fn):
        return partial(fn, run_cfg=run_cfg, log_mgr=log_mgr)

    g = StateGraph(FormFillingState)

    g.add_node("greet",              _node(node_greet))
    g.add_node("get_input",          _node(node_get_user_input))
    g.add_node("detect_intent",      _node(node_detect_intent))
    g.add_node("intent_unclear",     _node(node_intent_unclear))
    g.add_node("load_form",          _node(node_load_form))
    g.add_node("generate_questions", _node(node_generate_questions))
    g.add_node("ask_question",       _node(node_ask_question))
    g.add_node("get_input_field",    _node(node_get_user_input))
    g.add_node("parse_answer",       _node(node_parse_answer))
    g.add_node("extract",            _node(node_extract_information))
    g.add_node("follow_up",          _node(node_follow_up_question))
    g.add_node("repeat",             _node(node_repeat_question))
    g.add_node("get_input_retry",    _node(node_get_user_input))
    g.add_node("fill_form",          _node(node_fill_form))
    g.add_node("generate_json",      _node(node_generate_json))
    g.add_node("metrics",            _node(node_calculate_metrics))
    g.add_node("score",              _node(node_score))

    g.set_entry_point("greet")

    g.add_edge("greet",              "get_input")
    g.add_edge("get_input",          "detect_intent")
    g.add_edge("intent_unclear",     "get_input")
    g.add_edge("load_form",          "generate_questions")
    g.add_edge("generate_questions", "ask_question")
    g.add_edge("ask_question",       "get_input_field")
    g.add_edge("get_input_field",    "parse_answer")
    g.add_edge("follow_up",          "get_input_retry")
    g.add_edge("repeat",             "get_input_retry")
    g.add_edge("get_input_retry",    "parse_answer")
    g.add_edge("fill_form",          "generate_json")
    g.add_edge("generate_json",      "metrics")
    g.add_edge("metrics",            "score")
    g.add_edge("score",              END)

    g.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {"found": "load_form", "undefined": "intent_unclear", "max_attempts": END},
    )

    g.add_conditional_edges(
        "parse_answer",
        route_parse_action,
        {"extract": "extract", "follow_up": "follow_up", "repeat": "repeat"},
    )

    g.add_conditional_edges(
        "extract",
        route_next_field_or_done,
        {"next_field": "ask_question", "done": "fill_form"},
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
    run_log.respostas_formulario                   = filled
    run_log.contexto_final                         = final_state.get("context", [])
    run_log.report                                 = final_state.get("report", {})
    run_log.erros                                  = final_state.get("errors", [])
    run_log.architecture                           = run_cfg.architecture.value
    run_log.auto_user_enabled                      = run_cfg.auto_user
    run_log.auto_user_model                        = run_cfg.auto_user_model if run_cfg.auto_user else None
    run_log.models_per_agent = {
        "intent_agent":                              run_cfg.model_for_agent("intent_agent"),
        "interface_agent":                           run_cfg.model_for_agent("interface_agent"),
        "validation_agent":                          run_cfg.model_for_agent("validation_agent"),
        "output_generator":                          run_cfg.model_for_agent("output_generator"),
        "score_agent":                               run_cfg.model_for_agent("score_agent"),
        "answer_parsing_agent":                      run_cfg.model_for_agent("answer_parsing_agent"),
        "follow_up_question_agent":                  run_cfg.model_for_agent("follow_up_question_agent"),
        "form_filling_agent":                        run_cfg.model_for_agent("form_filling_agent"),
        "information_extraction_agent":              run_cfg.model_for_agent("information_extraction_agent"),
        "json_generation_agent":                     run_cfg.model_for_agent("json_generation_agent"),
        "question_generation_agent":                 run_cfg.model_for_agent("question_generation_agent"),
        "repeat_question_agent":                     run_cfg.model_for_agent("repeat_question_agent"),
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