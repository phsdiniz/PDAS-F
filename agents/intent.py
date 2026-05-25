"""
agents/intent.py — IntentAgent

Determina que formulário o utilizador quer preencher.
Devolve {"intent": "1"…"5"} ou {"intent": "UNDEFINED"}.
"""

from __future__ import annotations
from typing import Any

from agents.base import AgentResult, run_agent
from config import RunConfig

AGENT_NAME = "intent_agent"


def call_intent_agent(
    context: list[dict[str, str]],
    run_cfg: RunConfig,
    use_full_context: bool = True,
) -> AgentResult:
    """
    Args:
        context:          Histórico de mensagens [{"who": ..., "message": ...}, ...].
        run_cfg:          Configuração da run (modelo, arquitectura, etc.).
        use_full_context: Se True, envia todo o histórico; se False, apenas a última mensagem.
    """
    if use_full_context:
        text = "\n".join(f"{m['who']}: {m['message']}" for m in context)
    else:
        last = context[-1]
        text = f"{last['who']}: {last['message']}"

    return run_agent(
        prompt_file  = "IntentAgent.txt",
        replacements = {"[Inserir 1]": text},
        run_cfg      = run_cfg,
        agent_name   = AGENT_NAME,
    )