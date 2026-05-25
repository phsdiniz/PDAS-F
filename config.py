"""
config.py — Configuração centralizada do framework de benchmark MAS.

Todas as flags, modelos, caminhos e parâmetros dos experimentos vivem aqui.
Nada mais no código deve ter valores hardcoded — importa deste ficheiro.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import os

# ---------------------------------------------------------------------------
# Raiz do projeto
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent
DATA_DIR       = PROJECT_ROOT / "data"
FORMS_DIR      = DATA_DIR / "forms"
PROMPTS_DIR    = PROJECT_ROOT / "prompts"
PROMPTS_FF_MAP = PROMPTS_DIR / "FF_MAP"
PROMPTS_PDAS   = PROMPTS_DIR / "PDAS"
RESULTS_DIR    = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Modelos disponíveis
# ---------------------------------------------------------------------------
class Model(str, Enum):
    # --- OpenAI (cloud) ---
    GPT5       = "gpt-5"
    GPT5_MINI  = "gpt-5-mini"
    GPT5_NANO  = "gpt-5-nano"
    GPT4O      = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"

    # --- Locais via Ollama (prefixo "ollama/") ---
    # Instala com: ollama pull <nome>
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
        """Nome do modelo para passar ao cliente Ollama (sem prefixo)."""
        return self.value.replace("ollama/", "", 1)


# ---------------------------------------------------------------------------
# Arquitecturas disponíveis
# ---------------------------------------------------------------------------
class Architecture(str, Enum):
    VANILLA = "vanilla"
    FF_MAP  = "FF_MAP"
    PDAS    = "PDAS"
    PDAS_F  = "PDAS_F"


# ---------------------------------------------------------------------------
# Configuração por arquitectura
# ---------------------------------------------------------------------------
ARCHITECTURE_MODEL_DEFAULTS: dict[Architecture, Model] = {
    Architecture.VANILLA: Model.GPT5_NANO,
    Architecture.FF_MAP:  Model.GPT5_NANO,
    Architecture.PDAS:    Model.GPT5_NANO,
    Architecture.PDAS_F:  Model.GPT5_NANO,
}

AGENT_MODEL_OVERRIDES: dict[Architecture, dict[str, Optional[Model]]] = {
    Architecture.PDAS_F: {
        "planning_agent":    Model.QWEN3_30B,
        "evaluation_agent":  Model.QWEN3_30B,
        "interface_agent":   Model.LLAMA31_8B,
        "intent_agent":      Model.QWEN3_30B,
        "validation_agent":  Model.LLAMA31_8B,
        "single_task_agent": Model.LLAMA31_8B,
        "output_generator":  Model.QWEN3_30B,
        "score_agent":       Model.LLAMA31_8B,
    },
    Architecture.PDAS: {
        "planning_agent":    Model.QWEN3_30B,
        "evaluation_agent":  Model.QWEN3_30B,
        "interface_agent":   Model.LLAMA31_8B,
        "intent_agent":      Model.QWEN3_30B,
        "validation_agent":  Model.LLAMA31_8B,
        "single_task_agent": Model.LLAMA31_8B,
        "output_generator":  Model.QWEN3_30B,
        "score_agent":       Model.LLAMA31_8B,
    },
    Architecture.FF_MAP:  {},
    Architecture.VANILLA: {},
}


# ---------------------------------------------------------------------------
# Configuração de execução
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """
    Parâmetros de uma execução/simulação. Instanciado pelo CLI (main.py)
    ou directamente em scripts de experimento.
    """

    # --- Arquitectura e modelo ---
    architecture: Architecture    = Architecture.PDAS
    model_override: Optional[str] = None  # sobrescreve todos os agentes
    auto_user_model: str          = "gpt-4o"

    # --- Flags de comportamento ---
    debug_mode: bool       = False
    auto_user: bool        = False
    use_full_context: bool = True
    fake_validation: bool  = True

    # --- Batch ---
    num_simulations: int = 1

    # --- Limites e segurança ---
    max_intent_attempts: int     = 5
    max_validation_attempts: int = 3
    max_plan_stages: int         = 20

    # --- Reprodutibilidade ---
    # Se definido, o utilizador simulado é sempre escolhido de forma determinística.
    # Útil para comparar arquitecturas em condições controladas.
    auto_user_seed: Optional[int] = None

    # --- Construção do grafo com o graphviz ---
    viz: bool  = True

    # --- Mensagens fixas ---
    greeting_message: str = "Olá, como posso ajudá-lo?"

    def __post_init__(self) -> None:
        """Valida os campos após inicialização para falhar cedo e com mensagem clara."""
        if self.num_simulations < 1:
            raise ValueError("num_simulations deve ser >= 1")
        if self.max_intent_attempts < 1:
            raise ValueError("max_intent_attempts deve ser >= 1")
        if self.max_validation_attempts < 1:
            raise ValueError("max_validation_attempts deve ser >= 1")
        if self.max_plan_stages < 1:
            raise ValueError("max_plan_stages deve ser >= 1")

    def model_for_agent(self, agent_name: str) -> str:
        """
        Devolve o nome do modelo a usar para um agente específico.
        Prioridade: model_override > AGENT_MODEL_OVERRIDES > ARCHITECTURE_MODEL_DEFAULTS

        Aceita str directa (ex: "ollama/llama3.1:8b").
        O prefixo "ollama/" é sinal para o base.py usar o cliente Ollama.
        """
        if agent_name == "auto_user_agent":
            return self.auto_user_model

        if self.model_override:
            return self.model_override

        overrides   = AGENT_MODEL_OVERRIDES.get(self.architecture, {})
        agent_model = overrides.get(agent_name)
        if agent_model:
            # Os valores em AGENT_MODEL_OVERRIDES são sempre instâncias de Model
            return agent_model.value

        default = ARCHITECTURE_MODEL_DEFAULTS[self.architecture]
        return default.value


# ---------------------------------------------------------------------------
# Configuração de logging / Langfuse
# ---------------------------------------------------------------------------
@dataclass
class ObservabilityConfig:
    """
    Configurações do Langfuse.

    Setup:
        1. Criar conta em cloud.langfuse.com (plano gratuito)
        2. Settings → API Keys → criar par public/secret
        3. Definir variáveis de ambiente:
            export LANGFUSE_PUBLIC_KEY="pk-lf-..."
            export LANGFUSE_SECRET_KEY="sk-lf-..."
            export LANGFUSE_HOST="https://cloud.langfuse.com"   # EU
            # ou "https://us.cloud.langfuse.com"                # US
    """
    enabled: bool = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
    project_name: str = os.getenv("LANGFUSE_PROJECT", "mas-benchmark")

    def apply(self) -> None:
        """Desativa o LangSmith/LangChain tracing para evitar erros 403."""
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"]    = "false"

    def get_callback(self):
        """
        Devolve o CallbackHandler do Langfuse, ou None se não estiver configurado.
        Passar ao graph.invoke(): config={"callbacks": [handler]}
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
        """Verifica se as credenciais estão correctas antes de correr."""
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
# Instâncias prontas a importar
# ---------------------------------------------------------------------------
default_run   = RunConfig()
observability = ObservabilityConfig()


# ---------------------------------------------------------------------------
# Paleta de cores para terminal
# ---------------------------------------------------------------------------
class Colors:
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    RESET   = "\033[0m"