"""
agents/base.py — Chamada genérica ao OpenAI com resolução automática de modelo.

Todos os agentes específicos importam `call_llm` daqui.
Nunca instanciam o cliente OpenAI diretamente.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import os

import openai

from config import RunConfig, PROMPTS_DIR, PROMPTS_FF_MAP, PROMPTS_PDAS

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Clientes (singletons — inicializados uma vez)
# ---------------------------------------------------------------------------
def _get_openai_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # AVISO DE SEGURANÇA: api_secrets.py é um fallback de desenvolvimento local.
        # NUNCA faças commit deste ficheiro — adiciona-o ao .gitignore.
        # Em produção/CI usa sempre a variável de ambiente OPENAI_API_KEY.
        try:
            from api_secrets import api_key as _key  # type: ignore
            api_key = _key
        except ImportError:
            raise EnvironmentError(
                "OPENAI_API_KEY não definida. "
                "Define a variável de ambiente ou cria api_secrets.py com api_key='...' "
                "(NUNCA faças commit desse ficheiro)."
            )
    return openai.OpenAI(api_key=api_key)

'''
def _get_ollama_client() -> openai.OpenAI:
    """
    Ollama expõe uma API compatível com OpenAI em localhost:11434.
    Requer: ollama serve (correr em background antes de usar).
    Instalar: https://ollama.com  →  ollama pull llama3.1:8b
    """
    return openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama", 
    )
'''


_openai_client: openai.OpenAI | None = None
_ollama_client: openai.OpenAI | None = None
_hf_client:     openai.OpenAI | None = None


def get_client(model: str) -> openai.OpenAI:
    """Devolve o cliente correto com base no prefixo do modelo."""
    global _openai_client, _ollama_client, _hf_client

    if model.startswith("ollama/"):
        if _ollama_client is None:
            _ollama_client = openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # dummy key
            )
        return _ollama_client

    elif model.startswith("hf/"):
        if _hf_client is None:
            api_key = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
            if not api_key:
                raise ValueError("HF_API_KEY ou HF_TOKEN não definido. Define a variável de ambiente.")
            _hf_client = openai.OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=api_key,
            )
        return _hf_client

    else:
        if _openai_client is None:
            _openai_client = openai.OpenAI()
        return _openai_client


# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------
class AgentResult:
    """Resultado de uma chamada a um agente."""
    __slots__ = ("content", "input_tokens", "output_tokens", "tiktoken_input", "tiktoken_output")

    def __init__(self, content: str, input_tokens: int, output_tokens: int,
                 tiktoken_input: int, tiktoken_output: int):
        self.content            = content
        self.input_tokens       = input_tokens
        self.output_tokens      = output_tokens
        self.tiktoken_input     = tiktoken_input
        self.tiktoken_output    = tiktoken_output

    def __repr__(self) -> str:
        return (
            f"AgentResult(tokens_in={self.input_tokens}, "
            f"tokens_out={self.output_tokens}, content={self.content[:80]!r})"
        )

    def __iter__(self):
        return iter((self.content, self.input_tokens, self.output_tokens))


# ---------------------------------------------------------------------------
# Chamada base
# ---------------------------------------------------------------------------
def call_llm(
    prompt: str,
    model: str,
    system: str | None = None,
) -> AgentResult:
    """
    Chama o modelo e devolve um AgentResult.
    Suporta modelos OpenAI (cloud) e Ollama (local, prefixo "ollama/").

    Args:
        prompt: Conteúdo da mensagem do utilizador.
        model:  Nome do modelo. Exemplos:
                  "gpt-4o"               → OpenAI cloud
                  "ollama/llama3.1:8b"   → Ollama local
        system: System prompt opcional.
    """
    client     = get_client(model)
    model_name = (
        model.replace("ollama/", "", 1) if model.startswith("ollama/")
        else model.replace("hf/", "", 1) if model.startswith("hf/")
        else model
    )

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    content       = response.choices[0].message.content
    usage         = response.usage
    input_tokens  = usage.prompt_tokens     if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    tik_in        = len(enc.encode(system or "" + prompt))
    tik_out       = len(enc.encode(content))

    return AgentResult(
        content       = response.choices[0].message.content,
        input_tokens  = input_tokens,
        output_tokens = output_tokens,
        tiktoken_input=tik_in,
        tiktoken_output=tik_out,
    )


# ---------------------------------------------------------------------------
# Carregamento de prompts
# ---------------------------------------------------------------------------
def load_prompt(filename: str) -> str:
    """
    Carrega um ficheiro de prompt.

    Resolução de caminhos (por ordem de prioridade):
      1. Se filename começa com "FF_MAP/" → PROMPTS_FF_MAP
      2. Se filename começa com "PDAS/"   → PROMPTS_PDAS
      3. Caso contrário                   → PROMPTS_DIR (prompts partilhados)
    """
    if filename.startswith("FF_MAP/"):
        path = PROMPTS_FF_MAP / filename[len("FF_MAP/"):]
    elif filename.startswith("PDAS/"):
        path = PROMPTS_PDAS / filename[len("PDAS/"):]
    else:
        path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt não encontrado: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Agente genérico com substituição de placeholders
# ---------------------------------------------------------------------------
def run_agent(
    prompt_file: str,
    replacements: dict[str, Any],
    run_cfg: RunConfig,
    agent_name: str,
) -> AgentResult:
    """
    Carrega o prompt, substitui os placeholders e chama o modelo correto
    para este agente/arquitectura (resolvido via RunConfig).

    Args:
        prompt_file:  Nome do ficheiro em prompts/ (ex: "IntentAgent.txt").
        replacements: Mapeamento de placeholders → valores.
        run_cfg:      Configuração da run atual.
        agent_name:   Nome do agente (usado para resolver o modelo via config).
    """
    prompt = load_prompt(prompt_file)
    for key, value in replacements.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        prompt = prompt.replace(key, value)

    model = run_cfg.model_for_agent(agent_name)
    return call_llm(prompt, model=model)


# ---------------------------------------------------------------------------
# Contagem de tokens normalizada
# ---------------------------------------------------------------------------
def _tok(result: AgentResult) -> dict:
    """Delta nativo da API (custo real)."""
    return {
        "total_input_tokens": result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def _tok_multi(*results: AgentResult) -> dict:
    """Soma deltas de múltiplas chamadas."""
    in_total = sum(r.input_tokens for r in results)
    out_total = sum(r.output_tokens for r in results)
    return {
        "total_input_tokens": in_total,
        "total_output_tokens": out_total,
    }


def _tiktoken_approx(prompt: str, output: str) -> dict:
    """Contagem normalizada usando tokenizer cl100k_base (GPT-4o style)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return {
        "tiktoken_input": len(enc.encode(prompt)),
        "tiktoken_output": len(enc.encode(output)),
    }