"""
agents/interface.py — InterfaceAgent

Adapta uma instrução técnica ao estilo de linguagem do utilizador,
gerando a mensagem final exibida no chatbot.
"""

from __future__ import annotations

from agents.base import AgentResult, run_agent
from config import RunConfig

AGENT_NAME = "interface_agent"


def call_interface_agent(
    context: list[dict[str, str]],
    instruction: str,
    run_cfg: RunConfig,
    use_full_context: bool = True,
) -> AgentResult:
    """
    Args:
        context:          Histórico de mensagens.
        instruction:      Mensagem técnica a adaptar (ex: output do SingleTaskAgent).
        run_cfg:          Configuração da run.
        use_full_context: Se True, envia todo o histórico; se False, apenas a última mensagem.
    """
    if use_full_context:
        context_text = "\n".join(f"{m['who']}: {m['message']}" for m in context)
    else:
        last = context[-1]
        context_text = f"{last['who']}: {last['message']}"

    return run_agent(
        prompt_file  = "InterfaceAgent.txt",
        replacements = {
            "[Inserir 1]": context_text,
            "[Inserir 2]": instruction,
        },
        run_cfg      = run_cfg,
        agent_name   = AGENT_NAME,
    )