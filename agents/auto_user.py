"""
agents/auto_user.py — AutoUserAgent

Simula o utilizador humano para runs automáticas (auto_user=True).
Gera uma resposta plausível à última mensagem do chatbot,
com base no perfil de utilizador carregado de auto_users.txt.
"""

from __future__ import annotations
import random

from agents.base import AgentResult, call_llm
from config import RunConfig, DATA_DIR


def load_random_user_profile(seed: int | None = None) -> str:
    """
    Carrega um perfil aleatório de data/auto_users.txt.

    Args:
        seed: Se fornecido, a escolha é determinística — útil para reproduzir
              experiências em condições controladas. Se None, escolha aleatória.
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
    Gera a próxima mensagem do utilizador simulado.

    Args:
        user_profile:      Descrição textual do perfil (ex: "Maria, 72 anos, Lisboa...").
        last_bot_message:  Última mensagem do chatbot.
        run_cfg:           Configuração da run (para resolver o modelo).
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