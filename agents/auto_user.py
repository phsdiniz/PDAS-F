"""
agents/auto_user.py — AutoUserAgent

Simulates the human user for automatic runs (auto_user=True).
Generates a plausible reply to the chatbot's last message, based on the
user profile loaded from auto_users.txt.
"""

from __future__ import annotations
import random

from agents.base import AgentResult, call_llm
from config import RunConfig, DATA_DIR


def load_random_user_profile(seed: int | None = None) -> str:
    """
    Loads a random profile from data/auto_users.txt.

    Args:
        seed: If provided, the choice is deterministic — useful for
              reproducing experiments under controlled conditions.
              If None, the choice is random.
    """
    path  = DATA_DIR / "auto_users.txt"
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"auto_users.txt está vazio ou não encontrado em {path}")
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(lines)


def call_auto_user_agent(
    user_profile: str,
    last_bot_message: str,
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Generates the simulated user's next message.

    Args:
        user_profile:      Textual profile description (e.g. "Maria, 72 anos, Lisboa...").
        last_bot_message:  Chatbot's last message.
        run_cfg:           Run configuration (used to resolve the model).
    """
    prompt = (
        f"Contexto: \"{user_profile}\"\n"
        f"A partir do contexto, gere uma resposta para a última mensagem do chatbot: "
        f"\"{last_bot_message}\"\n"
        f"Você já pode ter mandado mensagens anteriores, então, para evitar repetir informações, "
        f"sua mensagem deve ser curta e focar **apenas** em responder a mensagem do chatbot! "
        f"Você pode criar dados para a resposta, desde que não fuja do contexto.\n"
    )
    model = run_cfg.model_for_agent("auto_user_agent")
    return call_llm(prompt, model=model)
