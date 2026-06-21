"""
agents/interface.py — InterfaceAgent

Adapts a technical instruction to the user's language style,
generating the final message displayed in the chatbot.
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
        context:          Message history.
        instruction:      Technical message to adapt (e.g. the SingleTaskAgent's output).
        run_cfg:          Run configuration.
        use_full_context: If True, sends the full history; if False, only the last message.
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
