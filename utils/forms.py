"""
utils/forms.py — Form I/O and quality-metric calculation.
"""

from __future__ import annotations
from pathlib import Path
import json
import re

from config import FORMS_DIR


# ---------------------------------------------------------------------------
# Loading and writing
# ---------------------------------------------------------------------------
def load_form(form_name: str) -> dict:
    """
    Loads a form JSON from data/forms/.

    Args:
        form_name: Name without extension (e.g. "formulario_2").
    """
    path = FORMS_DIR / f"{form_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Form not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_form(form_name: str, answers: dict, output_dir: Path) -> Path:
    """
    Writes the form's answers to a JSON file.

    Returns:
        Path of the generated file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{form_name}_preenchido.json"
    file_path.write_text(
        json.dumps({"form": answers}, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return file_path


# ---------------------------------------------------------------------------
# Quality metrics for the filled form
# ---------------------------------------------------------------------------
def hit_rate_calculation(json_output: dict) -> tuple[float, dict[str, str]]:
    """
    Calculates the rate of correctly filled required fields.

    Returns:
        (hit_rate, report) where:
            hit_rate — float between 0 and 1
            report   — dict {field_id: status message}
    """
    hit   = 0
    total = 0
    report: dict[str, str] = {}

    if not (isinstance(json_output, dict) and "secoes" in json_output):
        return 0.0, {"error": "Unrecognized form format (missing 'secoes')"}

    for secao in json_output["secoes"]:
        if not (isinstance(secao, dict) and "campos" in secao):
            continue
        for campo in secao["campos"]:
            if not isinstance(campo, dict):
                continue

            qid = campo.get("id", "unknown_field")
            _evaluate_campo(campo, report)

            if campo.get("required"):
                total += 1
                if report.get(qid, "").startswith("✔"):
                    hit += 1

    hit_rate = hit / total if total > 0 else 0.0
    return hit_rate, report


def _evaluate_campo(campo: dict, report: dict[str, str]) -> None:
    """Evaluates a single field and writes the result into report."""
    qid       = campo.get("id", "unknown_field")
    answer    = campo.get("value")
    required  = campo.get("required", False)
    tipo      = campo.get("type", "").lower()
    options   = campo.get("options", [])
    min_chars = campo.get("min_characters", 0) or 0

    _PLACEHOLDERS = (
        "nao fornecido", "nao preenchido", "not provided",
        "nenhum", "none", "n/a", "vazio", "empty",
    )
    if isinstance(answer, str) and any(p in answer.lower() for p in _PLACEHOLDERS):
        answer = None

    if answer is None:
        if required:
            report[qid] = "❌ Required field with no answer"
        else:
            report[qid] = "✔ Optional field with no answer (accepted)"
        return

    if tipo == "text":
        length = len(str(answer).strip())
        if min_chars > 0 and length < min_chars:
            report[qid] = f"❌ Answer too short: {length} chars (minimum {min_chars})"
        else:
            report[qid] = "✔ Field has an answer"

    elif tipo == "radio":
        opts_lower = [str(o).strip().lower() for o in options]
        if isinstance(answer, list):
            report[qid] = "❌ Radio field does not accept a list of answers"
        elif str(answer).strip().lower() in opts_lower:
            report[qid] = "✔ Answer matches one of the options"
        else:
            report[qid] = f"❌ Answer '{answer}' not in options {options}"

    elif tipo == "checkbox":
        opts_lower   = [str(o).strip().lower() for o in options]
        answers_list = answer if isinstance(answer, list) else [answer]
        invalid      = [a for a in answers_list if str(a).strip().lower() not in opts_lower]
        if invalid:
            report[qid] = f"❌ Invalid options selected: {invalid}"
        else:
            report[qid] = "✔ All answers within valid options"

    else:
        report[qid] = "✔ Field has an answer"


def preprocess_conditions(form: dict) -> tuple[list[str], dict]:
    """
    Pre-processes the form's visibility conditions.
    Returns:
    - visible_fields: list of IDs that are unconditionally visible
    - condition_map: {field_id: {"if": expr, "show": [ids]} }
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
                report[cid] = "✔ Conditional field not visible (accepted)"
                continue

            total += 1
            if value_in_filled is not None and value_in_filled != "":
                hit += 1
                report[cid] = f"✔ Answered: {value_in_filled}"
            else:
                report[cid] = "❌ Required field not answered"

    hit_rate = hit / total if total > 0 else 0.0

    return hit_rate, report


def calculate_advanced_metrics(
    state:          FormFillingState,
    original_form:  dict,
    filled_form:    dict,
    context:        list,
) -> dict:
    """
    Extra metrics useful for analysis.
    """
    metrics = {}

    # 1. Question efficiency = (number of required fields) ÷ (number of questions asked by the chatbot)
    # Ideal: > 1.0 (the higher, the more efficient)
    num_questions = sum(1 for m in context if m["who"] == "CHATBOT" and "?" in m["message"])
    num_required = sum(1 for s in original_form.get("secoes", []) for c in s.get("campos", []) if c.get("required", False))
    metrics["question_efficiency"] = num_required / max(num_questions, 1) if num_questions > 0 else 1.0

    # 2. Conditional compliance = (conditional fields correctly respected) ÷ (total conditional fields)
    # Ideal: 1.0 (100%)
    _, cond_report = evaluate_with_conditions(filled_form, original_form, state.get("form_answers", {}))
    cond_fields = list(preprocess_conditions(original_form)[1].keys())
    cond_success = sum(1 for cid in cond_fields if "✔" in cond_report.get(cid, ""))
    metrics["conditional_compliance"] = cond_success / max(len(cond_fields), 1) if cond_fields else 1.0

    # 3. Others can be added later
    metrics["num_chatbot_messages"] = len([m for m in context if m["who"] == "CHATBOT"])
    metrics["total_fields"] = sum(len(s.get("campos", [])) for s in original_form.get("secoes", []))

    return metrics


# ---------------------------------------------------------------------------
# Safe JSON parsing (used by every module)
# ---------------------------------------------------------------------------
def safe_json_loads(s: str) -> dict | list | None:
    """
    Tries to robustly parse a JSON string.

    Strategies, in order:
      1. Direct parse
      2. Extract a ```json ... ``` block (with or without surrounding text)
      3. Extract the longest JSON found in the string (greedy)
    """
    if not s or not isinstance(s, str):
        return None

    # 1. Direct parse
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Markdown block: ```json ... ``` with possible text before/after
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Extract the longest JSON (greedy) — catches JSON embedded in free text
    for pattern in (r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"):
        matches = re.findall(pattern, s)
        for m in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(m)
            except (json.JSONDecodeError, TypeError):
                pass

    return None
