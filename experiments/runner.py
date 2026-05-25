"""
experiments/runner.py — Runner de batch (N simulações × arquitectura × modelo).

Exemplo de uso programático:
    from config import RunConfig, Architecture
    from experiments.runner import run_batch

    run_batch(RunConfig(
        architecture    = Architecture.PDAS,
        num_simulations = 5,
        auto_user       = True,
        debug_mode      = False,
    ))
"""

from __future__ import annotations
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import csv
import json

from config import RunConfig, RESULTS_DIR, Architecture
from utils.logging import RunLog, color_print

_RUNNERS = {
    Architecture.VANILLA: "architectures.vanilla",
    Architecture.FF_MAP:  "architectures.FF_MAP",
    Architecture.PDAS:    "architectures.PDAS",
    Architecture.PDAS_F:  "architectures.PDAS_F",
}


def _get_runner(arch: Architecture):
    import importlib
    module = importlib.import_module(_RUNNERS[arch])
    return module.run


def _display_model(run_cfg: RunConfig) -> str:
    """
    Devolve um modelo representativo para exibir no log de progresso.
    Usa o agente mais pesado/relevante de cada arquitectura.
    Para arquitecturas sem planning_agent, cai no modelo default da arquitectura.
    """
    agent_map = {
        Architecture.PDAS:    "planning_agent",
        Architecture.PDAS_F:  "planning_agent",
        Architecture.FF_MAP:  "interface_agent",
        Architecture.VANILLA: "vanilla_agent",
    }
    agent = agent_map.get(run_cfg.architecture, "interface_agent")
    return run_cfg.model_for_agent(agent)


def run_batch(
    run_cfg: RunConfig,
    batch_dir: Path | None = None,
    langfuse_callback=None,
) -> list[dict]:
    """
    Executa N simulações com a configuração dada.

    Args:
        run_cfg:            Configuração da run (arquitectura, modelo, flags, N).
        batch_dir:          Directoria para guardar resultados. Se None, cria automaticamente.
        langfuse_callback:  CallbackHandler do Langfuse (ou None para desativar tracing).

    Returns:
        Lista de dicts de métricas (uma por simulação).
    """
    if batch_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        batch_dir = RESULTS_DIR / f"{run_cfg.architecture.value}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    runner  = _get_runner(run_cfg.architecture)
    results = []
    N       = run_cfg.num_simulations

    for i in range(N):
        color_print(
            f"Simulação {i+1}/{N} — {run_cfg.architecture.value} "
            f"[{_display_model(run_cfg)}]",
            "INFO",
        )
        sim_dir = batch_dir / f"sim_{i+1}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        with ExitStack() as stack:
            try:
                run_log: RunLog = runner(
                    stack, sim_dir, run_cfg,
                    langfuse_callback=langfuse_callback,
                )
                results.append(run_log.metricas.to_dict())
            except Exception as e:
                color_print(f"Simulação {i+1} falhou: {e}", "ERROR")
                try:
                    results.append({
                        **run_log.metricas.to_dict(),
                        "tiktoken_input_total":     run_log.metricas.total_tiktoken_input,
                        "tiktoken_output_total":    run_log.metricas.total_tiktoken_output,
                        "architecture":             run_log.architecture,
                        "auto_user_enabled":        run_log.auto_user_enabled,
                        "auto_user_model":          run_log.auto_user_model,
                        "models_per_agent":         json.dumps(run_log.models_per_agent, ensure_ascii=False),
                        **run_log.advanced_metrics,
                    })
                except Exception as e:
                    results.append({
                        "erro": str(e),
                    })

    if results:
        resumo_path = batch_dir / "resumo_simulacoes.csv"
        with open(resumo_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        color_print(f"\nBatch concluído → {resumo_path}", "INFO")

        json_path = batch_dir / "resumo_simulacoes.json"
        json_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return results
    