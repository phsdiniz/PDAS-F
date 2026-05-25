"""
agents/output_generator.py — OutputGeneratorAgent

Preenche o formulário JSON original com as respostas recolhidas do utilizador.
Devolve o formulário JSON com os campos "value" preenchidos.
"""

from __future__ import annotations

from agents.base import AgentResult, run_agent
from config import RunConfig

AGENT_NAME = "output_generator"


def call_output_generator_agent(
    form: dict,
    answers: dict,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Args:
        form:    Formulário JSON original (estrutura com "secoes" e "campos").
        answers: Dicionário de respostas recolhidas {stage_id: resposta_texto}.
        run_cfg: Configuração da run.
    """
    return run_agent(
        prompt_file  = "OutputGeneratorAgent.txt",
        replacements = {
            "[Inserir 1]": form,
            "[Inserir 2]": answers,
        },
        run_cfg      = run_cfg,
        agent_name   = AGENT_NAME,
    )