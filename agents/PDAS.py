"""
agents/PDAS.py — Agentes exclusivos das arquitecturas PDAS e PDAS_F.

Inclui: EvaluationAgent, PlanningAgent, PlanningAgentWithFeedback, SingleTaskAgent.

Estes agentes lêem os seus prompts de prompts/PDAS/.
"""

from __future__ import annotations

import json

from agents.base import AgentResult, run_agent
from config import RunConfig


def call_evaluation_agent(
    filled_form: dict,
    planning_output: str,
    hit_rate: float,
    iteration_count: int,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Analisa o formulário preenchido vs. o plano original e gera feedback
    estruturado (métricas + conselhos) para o PlanningAgent da próxima run.

    As métricas hit_rate e iteration_count são pré-calculadas pelo código
    e passadas directamente ao prompt para evitar que o modelo as calcule
    erroneamente a partir do JSON.

    Devolve JSON com: metricas, erros_por_campo, manter, ajustar, mudar.

    Args:
        filled_form:     Formulário preenchido (dict com "secoes" e "campos" com "value").
        planning_output: Plano original serializado (string JSON do PlanningAgent).
        hit_rate:        Taxa de campos obrigatórios correctamente preenchidos (0–1).
        iteration_count: Número de iterações/etapas executadas nesta run.
        run_cfg:         Configuração da run.
    """
    
    metrics_str = json.dumps(
        {"hit_rate": round(hit_rate, 4), "iteration_count": iteration_count},
        ensure_ascii=False,
    )
    return run_agent(
        prompt_file  = "PDAS/EvaluationAgent.txt",
        replacements = {
            "[Inserir 1]": filled_form,
            "[Inserir 2]": planning_output,
            "[Inserir 3]": metrics_str,
        },
        run_cfg      = run_cfg,
        agent_name   = "evaluation_agent",
    )


def call_planning_agent(
    form: dict,
    run_cfg: RunConfig,
    state: dict,
) -> AgentResult:
    """
    Gera o plano estratégico de etapas para preencher um formulário.
    Arquitectura PDAS (sem feedback).

    Devolve {"plan": [{"stage": N, "action": ..., "input": ..., "output": ...}, ...]}.

    Args:
        form:    Formulário JSON carregado.
        run_cfg: Configuração da run.
    """
    return run_agent(
        prompt_file  = "PDAS/PlanningAgent.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir condições]": json.dumps(state.get("condition_map", {}), ensure_ascii=False),
        },
        run_cfg      = run_cfg,
        agent_name   = "planning_agent",
    )


def call_planning_agent_with_feedback(
    form: dict,
    feedback: str,
    previous_plan: str,
    run_cfg: RunConfig,
    state: dict,
) -> AgentResult:
    """
    Gera o plano estratégico incorporando o feedback do EvaluationAgent da run anterior.
    Arquitectura PDAS_F (com feedback).

    Devolve {"plan": [{"stage": N, "action": ..., "input": ..., "output": ...}, ...]}.

    Args:
        form:          Formulário JSON carregado.
        feedback:      Output serializado do EvaluationAgent (str vazia se primeira run).
        previous_plan: Plano da run anterior (str vazia se primeira run).
        run_cfg:       Configuração da run.
    """
    return run_agent(
        prompt_file  = "PDAS/PlanningAgentWithFeedback.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir 2]": feedback      or "(sem feedback — primeira execução)",
            "[Inserir 3]": previous_plan or "(sem plano anterior — primeira execução)",
            "[Inserir condições]": json.dumps(state.get("condition_map", {}), ensure_ascii=False),
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
    Executa uma etapa individual do plano gerado pelo PlanningAgent.
    Devolve {"output": "..."}.

    Args:
        task_action:          Campo "action" da etapa do plano.
        task_input:           Input serializado para esta etapa (JSON string).
        task_expected_output: Campo "output" da etapa do plano (descrição do output esperado).
        run_cfg:              Configuração da run.
    """
    return run_agent(
        prompt_file  = "PDAS/SingleTaskAgent.txt",
        replacements = {
            "[Inserir 1]": task_action,
            "[Inserir 2]": task_input,
            "[Inserir 3]": task_expected_output,
        },
        run_cfg      = run_cfg,
        agent_name   = "single_task_agent",
    )