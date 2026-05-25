"""
agents/FF_MAP.py — Agentes exclusivos da arquitectura FF_MAP

Inclui: QuestionGeneration, AnswerParsing, InformationExtraction,
        FollowUpQuestion, RepeatQuestion, FormFilling, JsonGeneration.

Estes agentes lêem os seus prompts de prompts/FF_MAP/.
"""

from __future__ import annotations

from agents.base import AgentResult, run_agent
from config import RunConfig

# Prefixo da pasta de prompts desta arquitectura
_PREFIX = "FF_MAP/"


def call_question_generation_agent(form: dict, run_cfg: RunConfig) -> AgentResult:
    """Gera a lista de perguntas a fazer ao utilizador com base no formulário."""
    return run_agent(
        prompt_file  = _PREFIX + "question_generation.txt",
        replacements = {"[Inserir]": form},
        run_cfg      = run_cfg,
        agent_name   = "question_generation_agent",
    )


def call_answer_parsing_agent(
    dialogue: str,
    field_id: str,
    form: dict,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Decide a próxima ação: 'information_extraction', 'follow_up_question', ou 'repeat_question'.
    """
    return run_agent(
        prompt_file  = _PREFIX + "answer_parsing.txt",
        replacements = {
            "[Inserir 1]": dialogue,
            "[Inserir 2]": field_id,
            "[Inserir 3]": form,
        },
        run_cfg      = run_cfg,
        agent_name   = "answer_parsing_agent",
    )


def call_information_extraction_agent(dialogue: str, run_cfg: RunConfig) -> AgentResult:
    """Extrai a informação relevante de um diálogo pergunta/resposta."""
    return run_agent(
        prompt_file  = _PREFIX + "information_extraction.txt",
        replacements = {"[Inserir]": dialogue},
        run_cfg      = run_cfg,
        agent_name   = "information_extraction_agent",
    )


def call_follow_up_question_agent(
    dialogue: str,
    field_id: str,
    form: dict,
    run_cfg: RunConfig,
) -> AgentResult:
    """Gera uma pergunta de seguimento quando a resposta precisou de mais detalhe."""
    return run_agent(
        prompt_file  = _PREFIX + "follow_up_question.txt",
        replacements = {
            "[Inserir 1]": dialogue,
            "[Inserir 2]": field_id,
            "[Inserir 3]": form,
        },
        run_cfg      = run_cfg,
        agent_name   = "follow_up_question_agent",
    )


def call_repeat_question_agent(
    dialogue: str,
    field_id: str,
    form: dict,
    run_cfg: RunConfig,
) -> AgentResult:
    """Reformula a pergunta quando a resposta foi inválida ou incompreensível."""
    return run_agent(
        prompt_file  = _PREFIX + "repeat_question.txt",
        replacements = {
            "[Inserir 1]": dialogue,
            "[Inserir 2]": field_id,
            "[Inserir 3]": form,
        },
        run_cfg      = run_cfg,
        agent_name   = "repeat_question_agent",
    )


def call_form_filling_agent(
    form: dict,
    extracted_information: list,
    fields: list[str],
    run_cfg: RunConfig,
) -> AgentResult:
    """Mapeia a informação extraída para os campos do formulário."""
    return run_agent(
        prompt_file  = _PREFIX + "form_filling.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir 2]": extracted_information,
            "[Inserir 3]": fields,
        },
        run_cfg      = run_cfg,
        agent_name   = "form_filling_agent",
    )


def call_json_generation_agent(
    form: dict,
    filled_form: str,
    run_cfg: RunConfig,
) -> AgentResult:
    """Gera o formulário final preenchido em formato JSON válido."""
    return run_agent(
        prompt_file  = _PREFIX + "json_generation.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir 2]": filled_form,
        },
        run_cfg      = run_cfg,
        agent_name   = "json_generation_agent",
    )