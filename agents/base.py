"""
agents/base.py — Generic OpenAI call with automatic model resolution.

All specific agents import `call_llm` from here.
They never instantiate the OpenAI client directly.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import os

import openai

from config import RunConfig, PROMPTS_DIR

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Clients (singletons — initialized once)
# ---------------------------------------------------------------------------
def _get_openai_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # SECURITY WARNING: api_secrets.py is a local-development fallback.
        # NEVER commit this file — add it to .gitignore.
        # In production/CI always use the OPENAI_API_KEY environment variable.
        try:
            from api_secrets import api_key as _key  # type: ignore
            api_key = _key
        except ImportError:
            raise EnvironmentError(
                "OPENAI_API_KEY is not defined."
                "Set the environment variable or create api_secrets.py with api_key=‘...’ "
                "(NEVER commit this file)."
            )
    return openai.OpenAI(api_key=api_key)

'''
def _get_ollama_client() -> openai.OpenAI:
    """
    Ollama exposes an OpenAI-compatible API at localhost:11434.
    Requires: ollama serve (run in the background before use).
    Install: https://ollama.com  →  ollama pull llama3.1:8b
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
    """Returns the right client based on the model's prefix."""
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
# Return types
# ---------------------------------------------------------------------------
class AgentResult:
    """Result of a call to an agent."""
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
# Base call
# ---------------------------------------------------------------------------
def call_llm(
    prompt: str,
    model: str,
    system: str | None = None,
) -> AgentResult:
    """
    Calls the model and returns an AgentResult.
    Supports OpenAI (cloud) and Ollama (local, "ollama/" prefix) models.

    Args:
        prompt: User message content.
        model:  Model name. Examples:
                  "gpt-4o"               → OpenAI cloud
                  "ollama/llama3.1:8b"   → Ollama local
        system: Optional system prompt.
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
# Prompt loading
# ---------------------------------------------------------------------------
def load_prompt(filename: str) -> str:
    """
    Loads a prompt file from prompts/ (flat directory — every agent prompt
    lives there directly, regardless of architecture).
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt não encontrado: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Generic agent call with placeholder substitution
# ---------------------------------------------------------------------------
def run_agent(
    prompt_file: str,
    replacements: dict[str, Any],
    run_cfg: RunConfig,
    agent_name: str,
) -> AgentResult:
    """
    Loads the prompt, substitutes the placeholders, and calls the right
    model for this agent/architecture (resolved via RunConfig).

    Args:
        prompt_file:  File name in prompts/ (e.g. "PlanningAgent.txt").
        replacements: Mapping of placeholders → values.
        run_cfg:      Current run configuration.
        agent_name:   Agent name (used to resolve the model via config).
    """
    prompt = load_prompt(prompt_file)
    for key, value in replacements.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        prompt = prompt.replace(key, value)

    model = run_cfg.model_for_agent(agent_name)
    return call_llm(prompt, model=model)


# ---------------------------------------------------------------------------
# Normalized token counting
# ---------------------------------------------------------------------------
def _tok(result: AgentResult) -> dict:
    """Native API delta (actual cost)."""
    return {
        "total_input_tokens": result.input_tokens,
        "total_output_tokens": result.output_tokens,
    }


def _tok_multi(*results: AgentResult) -> dict:
    """Sums deltas across multiple calls."""
    in_total = sum(r.input_tokens for r in results)
    out_total = sum(r.output_tokens for r in results)
    return {
        "total_input_tokens": in_total,
        "total_output_tokens": out_total,
    }


def _tiktoken_approx(prompt: str, output: str) -> dict:
    """Normalized count using the cl100k_base tokenizer (GPT-4o style)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return {
        "tiktoken_input": len(enc.encode(prompt)),
        "tiktoken_output": len(enc.encode(output)),
    }
