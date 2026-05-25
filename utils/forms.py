"""
utils/forms.py — I/O de formulários e cálculo de métricas de qualidade.
"""

from __future__ import annotations
from pathlib import Path
import json
import re

from config import FORMS_DIR


# ---------------------------------------------------------------------------
# Carregamento e escrita
# ---------------------------------------------------------------------------
def load_form(form_name: str) -> dict:
    """
    Carrega um formulário JSON de data/forms/.

    Args:
        form_name: Nome sem extensão (ex: "formulario_2").
    """
    path = FORMS_DIR / f"{form_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Formulário não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_form(form_name: str, answers: dict, output_dir: Path) -> Path:
    """
    Escreve as respostas do formulário num ficheiro JSON.

    Returns:
        Caminho do ficheiro gerado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{form_name}_preenchido.json"
    file_path.write_text(
        json.dumps({"formulario": answers}, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return file_path


# ---------------------------------------------------------------------------
# Métricas de qualidade do formulário preenchido
# ---------------------------------------------------------------------------
def hit_rate_calculation(json_output: dict) -> tuple[float, dict[str, str]]:
    """
    Calcula a taxa de campos obrigatórios corretamente preenchidos.

    Returns:
        (hit_rate, report) onde:
            hit_rate — float entre 0 e 1
            report   — dict {campo_id: mensagem de estado}
    """
    hit   = 0
    total = 0
    report: dict[str, str] = {}

    if not (isinstance(json_output, dict) and "secoes" in json_output):
        return 0.0, {"erro": "Formato de formulário não reconhecido (falta 'secoes')"}

    for secao in json_output["secoes"]:
        if not (isinstance(secao, dict) and "campos" in secao):
            continue
        for campo in secao["campos"]:
            if not isinstance(campo, dict):
                continue

            qid = campo.get("id", "campo_desconhecido")
            _evaluate_campo(campo, report)

            if campo.get("required"):
                total += 1
                if report.get(qid, "").startswith("✔"):
                    hit += 1

    hit_rate = hit / total if total > 0 else 0.0
    return hit_rate, report


def _evaluate_campo(campo: dict, report: dict[str, str]) -> None:
    """Avalia um campo individualmente e escreve o resultado em report."""
    qid      = campo.get("id", "campo_desconhecido")
    answer   = campo.get("value")
    required = campo.get("required", False)
    tipo     = campo.get("type", "").lower()
    options  = campo.get("options", [])
    min_chars = campo.get("min_characters", 0) or 0

    _PLACEHOLDERS = (
        "nao fornecido", "nao preenchido", "not provided",
        "nenhum", "none", "n/a", "vazio", "empty",
    )
    if isinstance(answer, str) and any(p in answer.lower() for p in _PLACEHOLDERS):
        answer = None

    if answer is None:
        if required:
            report[qid] = "❌ Campo obrigatório sem resposta"
        else:
            report[qid] = "✔ Campo opcional sem resposta (aceito)"
        return

    if tipo == "text":
        length = len(str(answer).strip())
        if min_chars > 0 and length < min_chars:
            report[qid] = f"❌ Resposta com menos de {min_chars} caracteres (tem {length})"
        else:
            report[qid] = "✔ Campo com resposta"

    elif tipo == "radio":
        opts_lower = [str(o).strip().lower() for o in options]
        if isinstance(answer, list):
            report[qid] = "❌ Radio não aceita lista de respostas"
        elif str(answer).strip().lower() in opts_lower:
            report[qid] = "✔ Resposta dentro das opções"
        else:
            report[qid] = f"❌ Resposta '{answer}' fora das opções {options}"

    elif tipo == "checkbox":
        opts_lower   = [str(o).strip().lower() for o in options]
        answers_list = answer if isinstance(answer, list) else [answer]
        invalid      = [a for a in answers_list if str(a).strip().lower() not in opts_lower]
        if invalid:
            report[qid] = f"❌ Opções inválidas: {invalid}"
        else:
            report[qid] = "✔ Respostas dentro das opções"

    else:
        report[qid] = "✔ Campo com resposta"


def preprocess_conditions(form: dict) -> tuple[list[str], dict]:
    """
    Pré-processa as condições do formulário.
    Retorna:
    - visible_fields: lista de IDs que são visíveis incondicionalmente
    - condition_map: {campo_id: {"if": expr, "show": [ids]} }
    """
    visible_fields = []
    condition_map = {}

    for secao in form.get("secoes", []):
        for campo in secao.get("campos", []):
            cid = campo.get("id")
            if not cid:
                continue

            cond = campo.get("condition")
            if cond is None:
                visible_fields.append(cid)
            else:
                condition_map[cid] = {
                    "if": cond.get("if"),
                    "show": cond.get("show", [])
                }

    return visible_fields, condition_map


def evaluate_with_conditions(
    filled_form: dict,
    original_form: dict,
    form_answers: dict
) -> tuple[float, dict[str, str]]:
    hit = 0
    total = 0
    report = {}

    if "secoes" in filled_form:
        first_value = filled_form["secoes"][0]["campos"][0].get("value")

    filled_values = {}
    for secao in filled_form.get("secoes", []):
        for campo in secao.get("campos", []):
            cid = campo.get("id")
            if cid:
                filled_values[cid] = campo.get("value")

    for secao in original_form.get("secoes", []):
        for campo in secao.get("campos", []):
            cid = campo.get("id")
            if not cid:
                continue

            required = campo.get("required", False)
            value_in_filled = filled_values.get(cid) 

            is_visible = True 
            if not is_visible:
                report[cid] = "✔ Campo condicional não visível (aceite)"
                continue

            total += 1
            if value_in_filled is not None and value_in_filled != "":
                hit += 1
                report[cid] = f"✔ Respondido: {value_in_filled}"
            else:
                report[cid] = "❌ Obrigatório e não respondido"

    hit_rate = hit / total if total > 0 else 0.0

    return hit_rate, report


def calculate_advanced_metrics(
    state:          FormFillingState,
    original_form:  dict,
    filled_form:    dict,
    context:        list,
) -> dict:
    """
    Métricas extras úteis para análise.
    """
    metrics = {}

    # 1. Eficiência de perguntas = (número de campos obrigatórios) ÷ (número de perguntas feitas pelo chatbot)
    # Ideal: > 1.0 (quanto maior, mais eficiente)
    num_questions = sum(1 for m in context if m["who"] == "CHATBOT" and "?" in m["message"])
    num_required = sum(1 for s in original_form.get("secoes", []) for c in s.get("campos", []) if c.get("required", False))
    metrics["question_efficiency"] = num_required / max(num_questions, 1) if num_questions > 0 else 1.0

    # 2. Compliance condicional = (campos condicionais que foram corretamente respeitados) ÷ (total de campos condicionais)
    # Ideal: 1.0 (100%)
    _, cond_report = evaluate_with_conditions(filled_form, original_form, state.get("form_answers", {}))
    cond_fields = list(preprocess_conditions(original_form)[1].keys())
    cond_success = sum(1 for cid in cond_fields if "✔" in cond_report.get(cid, ""))
    metrics["conditional_compliance"] = cond_success / max(len(cond_fields), 1) if cond_fields else 1.0

    # 3. Outras que podes expandir depois
    metrics["num_chatbot_messages"] = len([m for m in context if m["who"] == "CHATBOT"])
    metrics["total_fields"] = sum(len(s.get("campos", [])) for s in original_form.get("secoes", []))

    return metrics


# ---------------------------------------------------------------------------
# Parse JSON seguro (usado por todos os módulos)
# ---------------------------------------------------------------------------
def safe_json_loads(s: str) -> dict | list | None:
    """
    Tenta fazer parse de uma string JSON de forma robusta.

    Estratégias por ordem:
      1. Parse directo
      2. Extracção de bloco ```json ... ``` (com ou sem texto antes)
      3. Extracção do JSON mais longo na string (greedy)
    """
    if not s or not isinstance(s, str):
        return None

    # 1. Parse directo
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Bloco markdown: ```json ... ``` com possível texto antes/depois
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Extrair JSON mais longo (greedy) — apanha JSONs no meio de texto livre
    for pattern in (r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"):
        matches = re.findall(pattern, s)
        for m in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(m)
            except (json.JSONDecodeError, TypeError):
                pass

    return None
