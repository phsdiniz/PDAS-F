"""
config.py — Centralized configuration for the MAS benchmark framework.

All flags, models, paths, and experiment parameters live here.
Nothing else in the code should have hardcoded values — import from this file.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import os

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent
DATA_DIR       = PROJECT_ROOT / "data"
FORMS_DIR      = DATA_DIR / "forms"
PROMPTS_DIR    = PROJECT_ROOT / "prompts"
RESULTS_DIR    = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------
class Model(str, Enum):
    # --- OpenAI (cloud) ---
    GPT5       = "gpt-5"
    GPT5_MINI  = "gpt-5-mini"
    GPT5_NANO  = "gpt-5-nano"
    GPT4O      = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"

    # --- Local via Ollama ("ollama/" prefix) ---
    # Install with: ollama pull <name>
    LLAMA31_8B  = "ollama/llama3.1:8b"
    LLAMA32_3B  = "ollama/llama3.2:3b"
    MISTRAL_7B  = "ollama/mistral:7b"
    QWEN25_7B   = "ollama/qwen2.5:7b"
    GPT_OSS_20B = "ollama/gpt-oss:20b"
    QWEN3_30B   = "ollama/qwen3:30b"

    # --- Hugging Face (cloud) ---
    HF_QWEN2_5_7B   = "hf/Qwen/Qwen2.5-7B-Instruct"
    HF_LLAMA3_1_8B  = "hf/meta-llama/Llama-3.1-8B-Instruct"
    HF_MISTRAL_7B   = "hf/mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
    HF_GPT_OSS_20B  = "hf/openai/gpt-oss-20b:groq"

    @property
    def is_local(self) -> bool:
        return self.value.startswith("ollama/")

    @property
    def ollama_model_name(self) -> str:
        """Model name to pass to the Ollama client (without the prefix)."""
        return self.value.replace("ollama/", "", 1)


# ---------------------------------------------------------------------------
# Available architectures
# ---------------------------------------------------------------------------
class Architecture(str, Enum):
    PDAS    = "PDAS"
    PDAS_F  = "PDAS_F"


# ---------------------------------------------------------------------------
# Per-architecture configuration
# ---------------------------------------------------------------------------
ARCHITECTURE_MODEL_DEFAULTS: dict[Architecture, Model] = {
    Architecture.PDAS:    Model.GPT5_NANO,
    Architecture.PDAS_F:  Model.GPT5_NANO,
}

AGENT_MODEL_OVERRIDES: dict[Architecture, dict[str, Optional[Model]]] = {
    Architecture.PDAS_F: {
        "planning_agent":    Model.QWEN3_30B,
        "evaluation_agent":  Model.QWEN3_30B,
        "interface_agent":   Model.LLAMA31_8B,
        "validation_agent":  Model.LLAMA31_8B,
        "single_task_agent": Model.LLAMA31_8B,
    },
    Architecture.PDAS: {
        "planning_agent":    Model.QWEN3_30B,
        "evaluation_agent":  Model.QWEN3_30B,
        "interface_agent":   Model.LLAMA31_8B,
        "validation_agent":  Model.LLAMA31_8B,
        "single_task_agent": Model.LLAMA31_8B,
    },
}


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """
    Parameters for a single run/simulation. Instantiated by the CLI
    (main.py) or directly in experiment scripts.
    """

    # --- Architecture and model ---
    architecture: Architecture    = Architecture.PDAS
    model_override: Optional[str] = None  # overrides every agent
    auto_user_model: str          = "gpt-4o"

    # --- Behaviour flags ---
    debug_mode: bool       = False
    auto_user: bool        = False
    use_full_context: bool = True
    fake_validation: bool  = True

    # --- Batch ---
    num_simulations: int = 1

    # --- Limits and safety nets ---
    # Max attempts for the Planning Agent's form-identification invocation
    # (see agents.PDAS.call_planning_agent_identify_form) before giving up.
    max_form_identification_attempts: int = 5
    max_validation_attempts: int = 3
    max_plan_stages: int         = 20

    # Maximum number of times the Planning Agent can be called to revise
    # the current stage after a NOT_VALIDATED from the Validation Agent on
    # a Task Agent's output. Once exceeded, the system records an explicit
    # failure state and proceeds with the last output generated
    # (see graphs/shared_nodes.node_escalate_output_validation).
    max_replanning_attempts: int = 2

    # hit_rate threshold that separates exploitation (refine the existing
    # plan, <10% changes) from exploration (explore new groupings/flows,
    # up to 20% changes) in PlanningAgentWithFeedback, and that bounds
    # when the EvaluationAgent may produce the "mudar" list.
    # Single source of truth — never hardcoded in the prompts.
    hit_rate_threshold: float = 0.7

    # --- Reproducibility ---
    # If set, the simulated user is always chosen deterministically.
    # Useful for comparing architectures under controlled conditions.
    auto_user_seed: Optional[int] = None

    # --- Graph rendering with graphviz ---
    viz: bool  = True

    # --- Fixed messages ---
    greeting_message: str = "Olá, como posso ajudá-lo?"

    def __post_init__(self) -> None:
        """Validate fields on init to fail fast with a clear message."""
        if self.num_simulations < 1:
            raise ValueError("num_simulations deve ser >= 1")
        if self.max_form_identification_attempts < 1:
            raise ValueError("max_form_identification_attempts deve ser >= 1")
        if self.max_validation_attempts < 1:
            raise ValueError("max_validation_attempts deve ser >= 1")
        if self.max_plan_stages < 1:
            raise ValueError("max_plan_stages deve ser >= 1")
        if self.max_replanning_attempts < 1:
            raise ValueError("max_replanning_attempts deve ser >= 1")
        if not (0.0 <= self.hit_rate_threshold <= 1.0):
            raise ValueError("hit_rate_threshold deve estar entre 0 e 1")

    def model_for_agent(self, agent_name: str) -> str:
        """
        Returns the model name to use for a specific agent.
        Priority: model_override > AGENT_MODEL_OVERRIDES > ARCHITECTURE_MODEL_DEFAULTS

        Accepts a plain string too (e.g. "ollama/llama3.1:8b").
        The "ollama/" prefix signals base.py to use the Ollama client.
        """
        if agent_name == "auto_user_agent":
            return self.auto_user_model

        if self.model_override:
            return self.model_override

        overrides   = AGENT_MODEL_OVERRIDES.get(self.architecture, {})
        agent_model = overrides.get(agent_name)
        if agent_model:
            # Values in AGENT_MODEL_OVERRIDES are always Model instances
            return agent_model.value

        default = ARCHITECTURE_MODEL_DEFAULTS[self.architecture]
        return default.value


# ---------------------------------------------------------------------------
# Logging / Langfuse configuration
# ---------------------------------------------------------------------------
@dataclass
class ObservabilityConfig:
    """
    Langfuse settings.

    Setup:
        1. Create an account at cloud.langfuse.com (free tier)
        2. Settings → API Keys → create a public/secret pair
        3. Set environment variables:
            export LANGFUSE_PUBLIC_KEY="pk-lf-..."
            export LANGFUSE_SECRET_KEY="sk-lf-..."
            export LANGFUSE_HOST="https://cloud.langfuse.com"   # EU
            # or "https://us.cloud.langfuse.com"                # US
    """
    enabled: bool = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
    project_name: str = os.getenv("LANGFUSE_PROJECT", "mas-benchmark")

    def apply(self) -> None:
        """Disable LangSmith/LangChain tracing to avoid 403 errors."""
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"]    = "false"

    def get_callback(self):
        """
        Returns the Langfuse CallbackHandler, or None if not configured.
        Pass to graph.invoke(): config={"callbacks": [handler]}
        """
        if not self.enabled:
            return None
        try:
            from langfuse.langchain import CallbackHandler
            return CallbackHandler()
        except ImportError:
            print("[WARN] langfuse não instalado. Corre: pip install langfuse")
            return None

    def verify_connection(self) -> bool:
        """Check that the credentials are correct before running."""
        if not self.enabled:
            return False
        try:
            from langfuse import get_client
            client = get_client()
            ok = client.auth_check()
            if ok:
                print("[INFO] Langfuse conectado com sucesso")
            else:
                print("[WARN] Langfuse auth_check falhou — verifica as chaves")
            return ok
        except Exception as e:
            print(f"[WARN] Langfuse não disponível: {e}")
            return False


# ---------------------------------------------------------------------------
# Ready-to-import instances
# ---------------------------------------------------------------------------
default_run   = RunConfig()
observability = ObservabilityConfig()


# ---------------------------------------------------------------------------
# Terminal color palette
# ---------------------------------------------------------------------------
class Colors:
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    RESET   = "\033[0m"
