"""
agents/validation.py — ValidationAgent

Valida se a última mensagem do utilizador é adequada e tem intenção clara.
Devolve {"status": "VALIDATED"} ou {"status": "NOT_VALIDATED", "reason": "..."}.
"""

from __future__ import annotations

from agents.base import AgentResult, run_agent
from config import RunConfig

AGENT_NAME = "validation_agent"

# Resposta fake usada quando fake_validation=True (evita chamadas desnecessárias à API)
_FAKE_VALIDATED = AgentResult(
    content       = '{"status": "VALIDATED"}',
    input_tokens  = 0,
    output_tokens = 0,
    tiktoken_input=0,
    tiktoken_output=0,
)


def call_validation_agent(
    context: list[dict[str, str]],
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Args:
        context: Histórico de mensagens. O penúltimo é o contexto, o último é a mensagem a validar.
        run_cfg: Configuração da run. Se run_cfg.fake_validation=True, devolve VALIDATED sem API call.
    """
    if run_cfg.fake_validation:
        return _FAKE_VALIDATED

    if len(context) < 2:
        # Não há histórico suficiente para validar — aceitar por defeito
        return _FAKE_VALIDATED

    history_text = f"{context[-2]['who']}: {context[-2]['message']}"
    message_text = f"{context[-1]['who']}: {context[-1]['message']}"

    return run_agent(
        prompt_file  = "ValidationAgent.txt",
        replacements = {
            "[Inserir 1]": history_text,
            "[Inserir 2]": message_text,
        },
        run_cfg      = run_cfg,
        agent_name   = AGENT_NAME,
    )