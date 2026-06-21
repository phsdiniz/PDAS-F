"""
utils/logging.py — Structured logging and run metrics.

Centralizes all writing to log files so that the architecture modules
don't need to know where or how to write.
"""

from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO
import json

from config import Colors, Architecture


# ---------------------------------------------------------------------------
# Per-architecture log files
# The logs common to every architecture are always opened.
# Architecture-specific ones are only opened for the relevant architecture.
# ---------------------------------------------------------------------------
_COMMON_LOG_FILES = [
    "geral", "chat", "metricas",
    "interface_agent", "validation_agent",
]

_ARCH_LOG_FILES: dict[str, list[str]] = {
    Architecture.PDAS.value: [
        "planning_agent", "single_task_agent",
    ],
    Architecture.PDAS_F.value: [
        "planning_agent", "single_task_agent", "evaluation_agent",
    ],
}


# ---------------------------------------------------------------------------
# Metrics structure
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    total_input_tokens:            int   = 0
    total_output_tokens:           int   = 0
    total_tiktoken_input:          int = 0
    total_tiktoken_output:         int = 0
    tempo_total_execucao_segundos: float = 0.0
    numero_iteracoes:              int   = 0
    hit_rate:                      float = 0.0

    def update_tokens(self, in_tokens: int, out_tokens: int) -> None:
        self.total_input_tokens  += in_tokens
        self.total_output_tokens += out_tokens

    def to_dict(self) -> dict:
        return {
            "total_input_tokens":            self.total_input_tokens,
            "total_output_tokens":           self.total_output_tokens,
            "total_tiktoken_input":          self.total_tiktoken_input,
            "total_tiktoken_output":         self.total_tiktoken_output,
            "tempo_total_execucao_segundos": round(self.tempo_total_execucao_segundos, 2),
            "numero_iteracoes":              self.numero_iteracoes,
            "hit_rate":                      round(self.hit_rate, 2),
        }


# ---------------------------------------------------------------------------
# State of a single run
# ---------------------------------------------------------------------------
@dataclass
class RunLog:
    metricas:             Metrics    = field(default_factory=Metrics)
    architecture:         str | None = None
    models_per_agent:     dict[str, str] | None = None          
    auto_user_enabled:    bool = False
    auto_user_model:      str | None = None
    advanced_metrics:     dict = field(default_factory=dict)
    user_profile:         str | None = None
    form_id_detectado:    str | None = None
    plano_inicial:        list | None = None
    respostas_parciais:   list        = field(default_factory=list)
    respostas_formulario: dict        = field(default_factory=dict)
    contexto_final:       list        = field(default_factory=list)
    agent_io:             dict        = field(default_factory=dict)
    output:               Any         = None
    report:               dict        = field(default_factory=dict)
    erros:                list[str]   = field(default_factory=list)

    def record_agent_io(self, agent_name: str, input_data: Any, output_data: Any) -> None:
        self.agent_io.setdefault(agent_name, []).append(
            {"input": input_data, "output": output_data}
        )

    def to_dict(self) -> dict:
        return {
            "user_profile":          self.user_profile,
            "form_id_detectado":     self.form_id_detectado,
            "plano_inicial":         self.plano_inicial,
            "respostas_parciais":    self.respostas_parciais,
            "respostas_formulario":  self.respostas_formulario,
            "contexto_final":        self.contexto_final,
            "agent_io":              self.agent_io,
            "output":                self.output,
            "report":                self.report,
            "erros":                 self.erros,
            "metricas":              self.metricas.to_dict(),
            "architecture":          self.architecture,
            "models_per_agent":      self.models_per_agent,
            "auto_user_enabled":     self.auto_user_enabled,
            "auto_user_model":       self.auto_user_model,
            "advanced_metrics":      self.advanced_metrics or {},
        }


# ---------------------------------------------------------------------------
# Log file manager for a run
# ---------------------------------------------------------------------------
class LogFileManager:
    """
    Opens only the log files relevant to the architecture in use.
    Uses ExitStack to guarantee everything is closed even on error.
    """
    def __init__(self, run_dir: Path, stack: ExitStack, architecture: str | None = None):
        self._files: dict[str, TextIO] = {}
        self._run_dir = run_dir
        self._stack = stack
        self._architecture = architecture

        for name in _COMMON_LOG_FILES:
            self._open_log(name)

        if architecture and architecture in _ARCH_LOG_FILES:
            self._arch_specific = set(_ARCH_LOG_FILES[architecture])
        else:
            self._arch_specific = set()
    
    def _open_log(self, name: str) -> None:
        if name not in self._files:
            path = self._run_dir / f"{name}.log"
            self._files[name] = self._stack.enter_context(open(path, "w", encoding="utf-8"))

    def write(self, name: str, content: str) -> None:
        """Writes to a log file; silently ignores it if it doesn't exist."""
        f = self._files.get(name)
        if f:
            f.write(content)
            f.flush()

    def record(self, agent_name: str, input_data: Any, output_data: str, run_log: RunLog | None = None) -> None:
        """
        Records an agent's IO in the log files.
        run_log is optional — if provided, also records it in memory.
        """
        if agent_name in self._arch_specific:
            self._open_log(agent_name)
        if run_log is not None:
            run_log.record_agent_io(agent_name, input_data, output_data)
        self.write("geral", f"[INFO] Chamando {agent_name}\n")
        self.write(agent_name, f"{output_data}\n\n")

    def save_metrics(self, run_log: RunLog) -> None:
        m = run_log.metricas
        self.write(
            "metricas",
            f"total_input_tokens={m.total_input_tokens}\n"
            f"total_output_tokens={m.total_output_tokens}\n"
            f"tempo_total_execucao_segundos={m.tempo_total_execucao_segundos:.2f}\n"
            f"numero_iteracoes={m.numero_iteracoes}\n"
            f"hit_rate={m.hit_rate}\n",
        )

    def save_json_log(self, run_log: RunLog, run_dir: Path) -> None:
        path = run_dir / "log_experimento.json"
        path.write_text(
            json.dumps(run_log.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Colored terminal output
# ---------------------------------------------------------------------------
def color_print(text: str, who: str, debug_mode: bool = True) -> None:
    """
    Prints colored text to the terminal.
    If who == "DEBUG" and debug_mode == False, prints nothing.
    """
    if who == "DEBUG" and not debug_mode:
        return

    mapping = {
        "CHATBOT": Colors.GREEN,
        "CITIZEN": Colors.BLUE,
        "DEBUG":   Colors.YELLOW,
        "ERROR":   Colors.RED,
        "INFO":    Colors.MAGENTA,
    }
    color = mapping.get(who, Colors.RESET)

    prefix = {
        "CHATBOT": "CHATBOT: ",
        "CITIZEN": "CIDADÃO: ",
        "DEBUG":   "",
        "ERROR":   "[ERROR] ",
        "INFO":    "[INFO] ",
    }.get(who, "")

    print(f"{color}{prefix}{text}{Colors.RESET}")
