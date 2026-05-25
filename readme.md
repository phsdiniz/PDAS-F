# MAS Benchmark

## Centralized configuration (`config.py`)

All project configuration lives in `config.py`. You should never hardcode values
scattered across the codebase.

### What you can configure

#### 1. Model per architecture (default)
```python
ARCHITECTURE_MODEL_DEFAULTS = {
    Architecture.VANILLA: Model.GPT5_NANO,
    Architecture.FF_MAP:  Model.GPT5_NANO,
    Architecture.PDAS:    Model.GPT5_MINI,
    Architecture.PDAS_F:  Model.GPT5,
}
```

#### 2. Model per agent inside an architecture
```python
AGENT_MODEL_OVERRIDES = {
    Architecture.PDAS_F: {
        "planning_agent":   Model.GPT5,      # agente mais pesado → modelo maior
        "intent_agent":     Model.GPT5_NANO, # tarefa simples → modelo leve
        ...
    }
}
```

#### 3. Execution flags via `RunConfig`
```python
run = RunConfig(
    architecture    = Architecture.PDAS,
    model_override  = "gpt-5-mini",   # força um modelo para TODOS os agentes
    debug_mode      = True,
    auto_user       = True,
    auto_user_seed  = 42,             # reprodutibilidade: mesmo perfil em todas as runs
    num_simulations = 5,
)
```

#### 4. Langfuse (`ObservabilityConfig`)
Set environment variables before running:
```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```
`ObservabilityConfig` automatically detects them and enables tracing.

To disable via CLI:
```bash
python main.py --no-langfuse
```

---

## System dependencies

This project uses Graphviz for graph visualization (via `pygraphviz` / `pydot`).

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y graphviz graphviz-dev
pip install pygraphviz
```

### macOS (brew)

```bash
brew install graphviz
pip install pygraphviz
```

### Windows
1. Download and install Graphviz from the official website:
https://graphviz.org/download/
2. Make sure to add Graphviz to your system PATH, for example:
```bash
C:\Program Files\Graphviz\bin
```
3. Then install pygraphviz in your Python environment:
```bash
pip install pygraphviz
```

### Then install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Note:
> `pygraphviz` is not a pure Python package — it depends on Graphviz C libraries.
> If you don’t install `graphviz-dev`, you may see errors like:
>
> `fatal error: graphviz/cgraph.h: No such file or directory`
>
> If you don’t need advanced graph features, you can use pydot `pydot` instead.
> `pygraphviz` is only required for advanced layout control.

---

## Folder structure

```
app/
├── config.py              ← EVERYTHING starts here
├── data/
│   ├── forms/             ← formulario_*.json  (1–5)
│   └── auto_users.txt     ← simulated user profiles (50 profiles)
├── prompts/               ← shared agent *.txt prompts
│   ├── FF_MAP/            ← FF_MAP architecture
│   └── PDAS/              ← PDAS / PDAS_F architectures
├── agents/                ← one file per agent
├── architectures/         ← one file per architecture (LangGraph graphs)
│   ├── vanilla.py
│   ├── FF_MAP.py
│   ├── PDAS.py
│   └── PDAS_F.py
├── graphs/                ← reusable LangGraph nodes across architectures
│   ├── state.py
│   └── shared_nodes.py
├── utils/                 ← logging, metrics, form I/O
├── experiments/           ← batch runner for N simulations
└── main.py                ← CLI entry point
```

---

## Available architectures

| CLI value | Class | Description |
|---|---|---|
| `vanilla` | `Architecture.VANILLA` | Monolithic agent — a single LLM handles the entire conversation |
| `FF_MAP` | `Architecture.FF_MAP` | Fixed network of specialized agents per field |
| `PDAS` | `Architecture.PDAS` | Planning + step-by-step execution, no inter-run feedback |
| `PDAS_F` | `Architecture.PDAS_F` | PDAS with EvaluationAgent feedback between runs |

---

## How to add a new model

1. Add the model to the `Model` enum in `config.py`
2. Define the default in `ARCHITECTURE_MODEL_DEFAULTS` (or let it inherit) 
3. No other files need to be changed

## How to add a new architecture

1. Add it to the `Architecture` enum
2. Define its default model in `ARCHITECTURE_MODEL_DEFAULTS`
3. Create the file in `architectures/nova_arquitectura.py`
4. Register it in `experiments/runner.py` inside the `_RUNNERS` dictionary

---

## Contributor notes

### Token counters in LangGraph state
`total_input_tokens` and `total_output_tokens` use an `operator.add` reducer.
Nodes must return only the delta of each call — LangGraph handles accumulation automatically:

```python
# CORRECT
return {
    "total_input_tokens":  result.input_tokens,   # delta desta chamada
    "total_output_tokens": result.output_tokens,  # delta desta chamada
}

# WRONG — causes double counting
return {
    "total_input_tokens": state.get("total_input_tokens", 0) + result.input_tokens,
}
```

### Simulated user profiles
Profiles live in `data/auto_users.txt` (one per line).
For reproducible benchmarks, use `--auto-user-seed <N>`.
Add diverse profiles to cover different forms and literacy levels.
