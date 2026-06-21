"""
main.py — CLI entry point.

Usage examples:
    # 5 simulations, PDAS, auto-user, debug
    python main.py -n 5 --mode PDAS --auto-user --debug

    # Force a single model for every agent
    python main.py --mode PDAS --model gpt-5

    # Planning with feedback, 10 runs (learns across runs)
    python main.py -n 10 --mode PDAS_F --auto-user

    # Change only the simulated user's model
    python main.py --mode PDAS --auto-user --auto-user-model gpt-4o-mini --model gpt-5-nano

    # Seed for reproducibility (same user profile across every run)
    python main.py -n 5 --mode PDAS --auto-user --auto-user-seed 42
"""

import argparse
from config import RunConfig, Architecture, ObservabilityConfig
from experiments.runner import run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAS Benchmark — Teste de arquitecturas de agentes para preenchimento de formulários"
    )

    parser.add_argument(
        "-n", "--num-simulacoes",
        type=int, default=1,
        help="Número de simulações (default=1)",
    )
    parser.add_argument(
        "--mode",
        choices=[a.value for a in Architecture],
        default=Architecture.PDAS.value,
        help="Arquitectura a testar",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Forçar um modelo para todos os agentes. "
            "Cloud: gpt-4o, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o-mini. "
            "Local (requer Ollama): ollama/llama3.1:8b, ollama/mistral:7b, ollama/qwen2.5:7b"
        ),
    )
    parser.add_argument(
        "--auto-user-model",
        default="gpt-4o",
        help=(
            "Modelo usado para simular o utilizador (--auto-user). "
            "Aceita os mesmos valores que --model. Default: gpt-4o"
        ),
    )
    parser.add_argument(
        "--auto-user-seed",
        type=int,
        default=None,
        help="Seed para seleção determinística do perfil de utilizador simulado.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa modo debug (imprime todos os outputs intermédios)",
    )
    parser.add_argument(
        "--auto-user",
        action="store_true",
        help="Simula o utilizador automaticamente via LLM",
    )
    parser.add_argument(
        "--no-full-context",
        action="store_true",
        help="Agentes usam apenas a última mensagem (em vez do histórico completo)",
    )
    parser.add_argument(
        "--real-validation",
        action="store_true",
        help="Activa o ValidationAgent real (por defeito usa fake=VALIDATED)",
    )
    parser.add_argument(
        "--max-replanning-attempts",
        type=int, default=2,
        help=(
            "Nº máximo de vezes que o Planning Agent revê a etapa atual "
            "depois de um NOT_VALIDATED do Validation Agent sobre o output "
            "do Task Agent, antes de escalar (default=2)."
        ),
    )
    parser.add_argument(
        "--hit-rate-threshold",
        type=float, default=0.7,
        help=(
            "Threshold de hit_rate que separa exploitation (<10%% de mudanças) "
            "de exploration (até 20%% de mudanças) no PlanningAgentWithFeedback "
            "(default=0.7)."
        ),
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Desativa o Langfuse mesmo que as variáveis de ambiente estejam definidas",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure observability
    obs = ObservabilityConfig()

    if args.no_langfuse:
        obs.enabled = False

    obs.apply()  # disable LangSmith/LangChain tracing

    if obs.enabled:
        obs.verify_connection()

    langfuse_callback = obs.get_callback()

    run_cfg = RunConfig(
        architecture             = Architecture(args.mode),
        model_override           = args.model,
        auto_user_model          = args.auto_user_model,
        auto_user_seed           = args.auto_user_seed,
        debug_mode               = args.debug,
        auto_user                = args.auto_user,
        use_full_context         = not args.no_full_context,
        fake_validation          = not args.real_validation,
        num_simulations          = args.num_simulacoes,
        max_replanning_attempts  = args.max_replanning_attempts,
        hit_rate_threshold       = args.hit_rate_threshold,
    )

    try:
        run_batch(run_cfg, langfuse_callback=langfuse_callback)
    except KeyboardInterrupt:
        print("\n[INFO] Execution interrupted by the user.")
    except Exception as e:
        print(f"[ERROR] Execution error: {e}")
        raise


if __name__ == "__main__":
    main()