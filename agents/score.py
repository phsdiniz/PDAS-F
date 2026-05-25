"""
agents/score.py — ScoreAgent

Atribui uma pontuação de 0 a 100 ao formulário preenchido,
com base na completude e validade das respostas.
Devolve apenas um número (ex: "87").
"""

from __future__ import annotations

from agents.base import AgentResult, run_agent
from config import RunConfig

AGENT_NAME = "score_agent"


def call_score_agent(
    filled_form: dict,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Args:
        filled_form: Formulário preenchido (dict).
        run_cfg:     Configuração da run.
    """
    return run_agent(
        prompt_file  = "ScoreAgent.txt",
        replacements = {"[Inserir 1]": filled_form},
        run_cfg      = run_cfg,
        agent_name   = AGENT_NAME,
    )