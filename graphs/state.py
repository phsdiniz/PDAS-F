"""
graphs/state.py — Estado partilhado entre todos os grafos LangGraph.

O LangGraph passa este estado entre nós. Cada nó devolve um dict parcial
com apenas os campos que alterou — o grafo faz o merge automaticamente.

NOTA SOBRE CONTADORES DE TOKENS:
  total_input_tokens e total_output_tokens usam Annotated[int, operator.add]
  como reducer. Isto significa que os nós devem devolver APENAS o delta
  (tokens desta chamada), não o total acumulado. O LangGraph soma
  automaticamente. Exemplo correto num nó:

      return {
          "total_input_tokens":  result.input_tokens,   # delta, não total
          "total_output_tokens": result.output_tokens,  # delta, não total
      }
"""

from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


# ---------------------------------------------------------------------------
# Estado principal da run
# ---------------------------------------------------------------------------
class FormFillingState(TypedDict, total=False):
    # --- Contexto da conversa ---
    context: list[dict]          # [{"who": "CHATBOT"|"CITIZEN", "message": "..."}]

    # --- Formulário ---
    form: dict
    form_name: str
    form_answers: dict           # {stage_id: resposta_texto} — respostas brutas
    filled_form: dict            # formulário JSON com "value" preenchidos

    # --- Intenção ---
    intent: str | None           # "1"…"5" ou "UNDEFINED" ou None

    # --- Plano (planning-centric) ---
    plan: list[dict]
    plan_str: str
    current_stage_index: int
    single_task_output: str

    # --- Feedback inter-runs (PDAS_F) ---
    feedback: str
    previous_plan: str

    # --- Contadores e flags ---
    intent_attempts: int
    validation_attempts: int
    iteration_count: int

    # --- Vanilla LLM ---
    vanilla_status: str          # "in_progress" ou "complete"

    # --- FF_MAP ---
    questions: list[str]
    fields: list[str]
    current_field_index: int
    extracted_info: list
    current_dialogue: str
    parse_action: str            # "information_extraction" | "follow_up_question" | "repeat_question"
    filled_form_text: str

    # --- Resultados finais ---
    hit_rate:           float
    report:             dict
    score:              str
    advanced_metrics:   dict

    # --- Logging ---
    # Acumulador de IO dos agentes (append-only)
    agent_io_log: Annotated[list[dict], operator.add]

    # Erros acumulados
    errors: Annotated[list[str], operator.add]

    # Contagem de tokens
    total_input_tokens:     Annotated[int, operator.add]
    total_output_tokens:    Annotated[int, operator.add]
    total_tiktoken_input:   Annotated[int, operator.add]
    total_tiktoken_output:  Annotated[int, operator.add]

    # --- Perfil do utilizador simulado ---
    user_profile: str | None


# ---------------------------------------------------------------------------
# Estado inicial vazio
# ---------------------------------------------------------------------------
def initial_state() -> FormFillingState:
    return FormFillingState(
        context              = [],
        form                 = {},
        form_name            = "",
        form_answers         = {},
        filled_form          = {},
        intent               = None,
        plan                 = [],
        plan_str             = "",
        current_stage_index  = 0,
        single_task_output   = "",
        feedback             = "",
        previous_plan        = "",
        intent_attempts      = 0,
        validation_attempts  = 0,
        iteration_count      = 0,
        vanilla_status       = "in_progress",
        questions            = [],
        fields               = [],
        current_field_index  = 0,
        extracted_info       = [],
        current_dialogue     = "",
        parse_action         = "",
        filled_form_text     = "",
        hit_rate             = 0.0,
        report               = {},
        score                = "",
        advanced_metrics     = {},
        agent_io_log         = [],
        errors               = [],
        total_input_tokens   = 0,
        total_output_tokens  = 0,
        user_profile         = None,
    )