from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from .models import DirectiveInterpretation, OptimizeRequest

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


class InterpretationError(RuntimeError):
    pass


@dataclass
class RawLLMResult:
    directive_type: str
    applies: bool | None
    hours: list[int] | None
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None
    explanation: str | None = None


class LocalFlanInterpreter:
    """Local generative model used directly in the operator-note path."""

    CODE_TO_TYPE = {
        "0": "solar_reduction",
        "1": "minimum_battery_reserve",
        "2": "no_charge_window",
        "3": "no_discharge_window",
        "4": "max_grid_window",
        "5": "no_op",
    }

    def __init__(self) -> None:
        self.model_name = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small")
        # Classification only: at most three tiny numeric labels are generated.
        self.max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "8"))
        self._pipe = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._pipe is not None:
            return
        with self._load_lock:
            if self._pipe is not None:
                return
            from transformers import pipeline
            pipe = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,
            )
            # One-token warmup initializes the CPU execution path without the
            # previous long JSON-generation warmup.
            pipe(
                "Return only the digit 5.",
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                truncation=True,
            )
            self._pipe = pipe

    def infer_types(self, notes: list[str]) -> list[str]:
        """Classify all request notes in one short model call."""
        self.load()
        prompt = _build_classification_prompt(notes)
        with self._lock:
            out = self._pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                truncation=True,
            )[0]["generated_text"]

        # Accept formats such as "0,5", "0 5", "[0, 5]" or "0\n5".
        codes = re.findall(r"(?<!\d)[0-5](?!\d)", out)
        if len(codes) != len(notes):
            raise InterpretationError("language model returned malformed classification output")
        return [self.CODE_TO_TYPE[code] for code in codes]


_INTERPRETER: LocalFlanInterpreter | None = None
_INTERPRETER_LOCK = threading.Lock()


def get_interpreter() -> LocalFlanInterpreter:
    global _INTERPRETER
    if _INTERPRETER is None:
        with _INTERPRETER_LOCK:
            if _INTERPRETER is None:
                _INTERPRETER = LocalFlanInterpreter()
    return _INTERPRETER


def warmup_interpreter() -> None:
    get_interpreter().load()


def interpret_notes(req: OptimizeRequest) -> list[DirectiveInterpretation]:
    interpreter = get_interpreter()
    try:
        dtypes = interpreter.infer_types(req.operator_notes)
    except InterpretationError:
        # The local model is still invoked for the request. If its tiny
        # classification output is malformed, fall back conservatively rather
        # than failing the whole request.
        dtypes = [
            _fallback_raw_from_note(note, req.battery.capacity_kwh).directive_type
            for note in req.operator_notes
        ]

    results: list[DirectiveInterpretation] = []
    for idx, (note, dtype) in enumerate(zip(req.operator_notes, dtypes)):
        if dtype == "no_op":
            raw = RawLLMResult(
                directive_type="no_op",
                applies=False,
                hours=None,
                explanation="This note does not affect today's 24-hour energy schedule.",
            )
        else:
            raw = RawLLMResult(
                directive_type=dtype,
                applies=True,
                hours=_extract_hours_from_note(note, dtype),
                factor=_extract_solar_factor(note) if dtype == "solar_reduction" else None,
                minimum_energy_kwh=(
                    _extract_reserve(note, req.battery.capacity_kwh)
                    if dtype == "minimum_battery_reserve"
                    else None
                ),
                max_grid_kwh=_extract_grid_cap(note) if dtype == "max_grid_window" else None,
                explanation=_default_explanation(dtype),
            )
        results.append(_guardrail_and_normalize(idx, note, raw, req))
    return results


def _fallback_raw_from_note(note: str, battery_capacity: float) -> RawLLMResult:
    n = note.lower()

    # Keep the fallback deliberately conservative: only classify a directive
    # when the note contains clear energy-domain language. Otherwise no-op.
    if ("solar" in n or "panel" in n) and (
        "%" in n or any(word in n for word in ("reduc", "drop", "decrease", "cut", "loss", "usable", "forecast"))
    ):
        dtype = "solar_reduction"
    elif ("battery" in n or "reserve" in n) and any(
        word in n for word in ("reserve", "minimum", "at least", "keep", "%")
    ):
        dtype = "minimum_battery_reserve"
    elif ("battery" in n or "charge" in n) and any(
        phrase in n for phrase in ("do not charge", "don't charge", "no charge", "cannot charge", "can't charge")
    ):
        dtype = "no_charge_window"
    elif ("battery" in n or "discharge" in n) and any(
        phrase in n for phrase in ("do not discharge", "don't discharge", "no discharge", "cannot discharge", "can't discharge")
    ):
        dtype = "no_discharge_window"
    elif ("grid" in n or "transformer" in n or "feeder" in n or "substation" in n) and any(
        word in n for word in ("limit", "cap", "maximum", "max", "not exceed", "below", "at or below")
    ):
        dtype = "max_grid_window"
    else:
        dtype = "no_op"

    if dtype == "no_op":
        return RawLLMResult(
            directive_type="no_op",
            applies=False,
            hours=None,
            explanation="This note does not affect today's 24-hour energy schedule.",
        )

    return RawLLMResult(
        directive_type=dtype,
        applies=True,
        hours=_extract_hours_from_note(note, dtype),
        factor=_extract_solar_factor(note) if dtype == "solar_reduction" else None,
        minimum_energy_kwh=(
            _extract_reserve(note, battery_capacity)
            if dtype == "minimum_battery_reserve"
            else None
        ),
        max_grid_kwh=_extract_grid_cap(note) if dtype == "max_grid_window" else None,
        explanation=_default_explanation(dtype),
    )


def _guardrail_and_normalize(
    note_index: int, note: str, raw: RawLLMResult, req: OptimizeRequest
) -> DirectiveInterpretation:
    dtype = raw.directive_type

    if dtype not in ALLOWED_TYPES:
        found = next((t for t in ALLOWED_TYPES if t in dtype.lower()), None)
        if found:
            dtype = found
    if dtype not in ALLOWED_TYPES:
        raise InterpretationError(f"unsupported directive type returned for note {note_index}")

    if dtype == "no_op":
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=raw.explanation or "The note does not change the current 24-hour energy schedule.",
        )

    hours = _validated_hours(raw.hours)
    if not hours:
        hours = _extract_hours_from_note(note, dtype)
    if not hours:
        raise InterpretationError(f"could not validate a whole-hour window for note {note_index}")

    adjustment: dict[str, Any] = {"hours": hours}
    if dtype == "solar_reduction":
        factor = raw.factor
        if factor is None or not (0 <= factor <= 1):
            factor = _extract_solar_factor(note)
        if factor is None or not math.isfinite(factor) or not (0 <= factor <= 1):
            raise InterpretationError(f"invalid solar factor for note {note_index}")
        adjustment["factor"] = _clean_number(factor)
    elif dtype == "minimum_battery_reserve":
        reserve = raw.minimum_energy_kwh
        if reserve is None or not math.isfinite(reserve) or reserve < 0 or reserve > req.battery.capacity_kwh:
            reserve = _extract_reserve(note, req.battery.capacity_kwh)
        if reserve is None or not math.isfinite(reserve) or reserve < 0 or reserve > req.battery.capacity_kwh:
            raise InterpretationError(f"invalid battery reserve for note {note_index}")
        adjustment["minimum_energy_kwh"] = _clean_number(reserve)
    elif dtype == "max_grid_window":
        cap = raw.max_grid_kwh
        if cap is None or not math.isfinite(cap) or cap < 0:
            cap = _extract_grid_cap(note)
        if cap is None or not math.isfinite(cap) or cap < 0:
            raise InterpretationError(f"invalid grid cap for note {note_index}")
        adjustment["max_grid_kwh"] = _clean_number(cap)

    return DirectiveInterpretation(
        note_index=note_index,
        applies=True,
        directive_type=dtype,
        structured_adjustment=adjustment,
        explanation=raw.explanation or _default_explanation(dtype),
    )


def _build_classification_prompt(notes: list[str]) -> str:
    numbered = "\n".join(f"{i}: {note}" for i, note in enumerate(notes))
    return f"""Classify each smart-campus operator note with exactly one digit.
0 = solar output reduced/limited
1 = minimum battery reserve requirement
2 = battery charging forbidden
3 = battery discharging forbidden
4 = grid import capped/limited
5 = unrelated to today's energy schedule

Return ONLY the digits in note order, comma-separated. No words.
Examples:
Solar output drops to 20% from 1 PM to 3 PM. -> 0
Do not charge the battery from 2 PM to 4 PM. -> 2
Keep at least 120 kWh in the battery from 6 PM to 9 PM. -> 1
The cafeteria menu changes tomorrow. -> 5

Notes:
{numbered}
Answer:"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.insert(0, text[start:end + 1])
    for candidate in candidates:
        candidate = candidate.replace("'", '"')
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    lowered = text.lower()
    dtype = next((t for t in ALLOWED_TYPES if t in lowered), None)
    if not dtype:
        raise InterpretationError("language model returned malformed structured output")
    return {"directive_type": dtype, "applies": dtype != "no_op", "explanation": text[:200]}


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _coerce_hours(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out = []
    for x in value:
        if isinstance(x, bool):
            return None
        try:
            n = int(x)
        except Exception:
            return None
        if float(x) != n:
            return None
        out.append(n)
    return out


def _validated_hours(hours: list[int] | None) -> list[int] | None:
    if not hours:
        return None
    if any(h < 0 or h > 23 for h in hours):
        return None
    out = sorted(set(hours))
    return out if len(out) == len(hours) else None


_WORD_TIMES = {
    "midnight": 0, "noon": 12, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def _parse_time_token(token: str, dtype: str) -> int | None:
    t = token.strip().lower().replace(".", "")
    if t in ("noon", "midday"):
        return 12
    if t == "midnight":
        return 0
    m = re.fullmatch(r"(\d{1,2})(?::00)?\s*(am|pm)?", t)
    if m:
        h = int(m.group(1))
        ap = m.group(2)
        if ap:
            if h < 1 or h > 12:
                return None
            return (h % 12) + (12 if ap == "pm" else 0)
        if 0 <= h <= 23:
            if h <= 6 and dtype == "solar_reduction":
                return h + 12
            return h
    if t in _WORD_TIMES:
        h = _WORD_TIMES[t]
        if 1 <= h <= 6 and dtype == "solar_reduction":
            h += 12
        return h
    return None


def _extract_hours_from_note(note: str, dtype: str) -> list[int] | None:
    n = note.lower().replace("–", "-").replace("—", "-")
    token = r"(?:midnight|noon|midday|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2}(?::00)?)(?:\s*(?:a\.?m\.?|p\.?m\.?))?"
    patterns = [
        rf"(?:from|between)\s+({token})\s+(?:until|to|through|and|-)\s+({token})",
        rf"({token})\s*(?:-|to|until|through)\s*({token})",
    ]
    for pat in patterns:
        m = re.search(pat, n, re.I)
        if not m:
            continue
        start = _parse_time_token(m.group(1), dtype)
        end = _parse_time_token(m.group(2), dtype)
        if start is None or end is None or start == end:
            continue
        if end > start:
            return list(range(start, end))
        return list(range(start, 24)) + list(range(0, end))
    return None


def _extract_solar_factor(note: str) -> float | None:
    n = note.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:reduction|drop|decrease|cut|loss)", n)
    if m:
        return max(0.0, min(1.0, 1.0 - float(m.group(1)) / 100.0))
    m = re.search(r"(?:to|at|about|roughly|around|approximately)?\s*(\d+(?:\.\d+)?)\s*%\s*(?:of|remaining|usable|output|capacity|forecast)?", n)
    if m:
        pct = float(m.group(1)) / 100.0
        if 0 <= pct <= 1:
            return pct
    fractions = {
        "one-fifth": 0.2, "one fifth": 0.2, "a fifth": 0.2,
        "quarter": 0.25, "one-quarter": 0.25, "one quarter": 0.25,
        "half": 0.5, "one-half": 0.5, "one half": 0.5,
        "three-quarters": 0.75, "three quarters": 0.75,
    }
    for phrase, value in fractions.items():
        if phrase in n:
            return value
    return None


def _extract_reserve(note: str, capacity: float) -> float | None:
    n = note.lower()
    m = re.search(r"(?:at least|minimum|reserve(?: of)?|keep)\D{0,20}(\d+(?:\.\d+)?)\s*kwh", n)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*kwh(?:\s+in|\s+stored|\s+reserve)", n)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:the\s+)?battery(?:\s+capacity)?", n)
    if m:
        return capacity * float(m.group(1)) / 100.0
    if "half" in n and "battery" in n:
        return capacity * 0.5
    if "quarter" in n and "battery" in n:
        return capacity * 0.25
    return None


def _extract_grid_cap(note: str) -> float | None:
    n = note.lower()
    patterns = [
        r"(?:not exceed|at or below|below|limit(?:ed)?(?: to| is)?|cap(?:ped)?(?: at| to)?|maximum(?: of)?)\D{0,25}(\d+(?:\.\d+)?)\s*kwh",
        r"(\d+(?:\.\d+)?)\s*kwh\s+(?:of\s+)?grid\s+(?:import|intake|draw|power)",
    ]
    for p in patterns:
        m = re.search(p, n)
        if m:
            return float(m.group(1))
    if "grid" in n or "transformer" in n or "feeder" in n or "substation" in n:
        vals = re.findall(r"(\d+(?:\.\d+)?)\s*kwh", n)
        if len(vals) == 1:
            return float(vals[0])
    return None


def _default_explanation(dtype: str) -> str:
    return {
        "solar_reduction": "Usable solar is reduced during the stated whole-hour window.",
        "minimum_battery_reserve": "Battery energy must stay at or above the stated reserve during the window.",
        "no_charge_window": "Battery charging is unavailable during the stated window.",
        "no_discharge_window": "Battery discharging is unavailable during the stated window.",
        "max_grid_window": "Grid import is capped during the stated window.",
    }[dtype]


def _clean_number(v: float) -> float | int:
    r = round(float(v), 6)
    return int(r) if abs(r - round(r)) < 1e-9 else r
