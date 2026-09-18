"""
app/validators.py — Deterministic guardrail validation for LLM directive output.

Validates the structured output from the LLM before passing it to the optimizer.
All checks are pure Python — no LLM calls.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _validate_hours(hours: Any) -> Tuple[bool, str]:
    """Validate that hours is a sorted list of unique integers in [0, 23]."""
    if not isinstance(hours, list) or len(hours) == 0:
        return False, "hours must be a non-empty list"
    for h in hours:
        if not isinstance(h, int) or h < 0 or h > 23:
            return False, f"invalid hour value: {h!r} (must be int 0-23)"
    if len(set(hours)) != len(hours):
        return False, "hours must be unique"
    if hours != sorted(hours):
        return False, "hours must be in ascending order"
    return True, ""


def _validate_directive_entry(entry: Dict, num_notes: int) -> Tuple[bool, str]:
    """Validate a single directive entry."""
    # Required fields
    for field in ("note_index", "applies", "directive_type", "explanation"):
        if field not in entry:
            return False, f"missing field: {field}"

    note_index = entry.get("note_index")
    if not isinstance(note_index, int) or note_index < 0 or note_index >= num_notes:
        return False, f"invalid note_index: {note_index}"

    directive_type = entry.get("directive_type")
    if directive_type not in VALID_DIRECTIVE_TYPES:
        return False, f"invalid directive_type: {directive_type!r}"

    applies = entry.get("applies")
    if not isinstance(applies, bool):
        return False, "applies must be a boolean"

    adj = entry.get("structured_adjustment")

    # no_op rules
    if directive_type == "no_op":
        if applies is not False:
            return False, "no_op must have applies=false"
        if adj is not None:
            return False, "no_op must have structured_adjustment=null"
        return True, ""

    # non-no_op rules
    if applies is not True:
        return False, f"non-no_op directive_type '{directive_type}' must have applies=true"

    if adj is None or not isinstance(adj, dict):
        return False, f"'{directive_type}' must have a structured_adjustment dict"

    # Hours present in adjustment
    if "hours" not in adj:
        return False, f"'{directive_type}' structured_adjustment missing 'hours'"

    ok, msg = _validate_hours(adj["hours"])
    if not ok:
        return False, f"'{directive_type}' hours invalid: {msg}"

    # Type-specific field checks
    if directive_type == "solar_reduction":
        factor = adj.get("factor")
        if not isinstance(factor, (int, float)) or not (0 < factor <= 1):
            return False, f"solar_reduction factor must be float in (0, 1], got {factor!r}"

    elif directive_type == "minimum_battery_reserve":
        min_e = adj.get("minimum_energy_kwh")
        if not isinstance(min_e, (int, float)) or min_e < 0:
            return False, f"minimum_battery_reserve minimum_energy_kwh must be >= 0, got {min_e!r}"

    elif directive_type == "max_grid_window":
        max_g = adj.get("max_grid_kwh")
        if not isinstance(max_g, (int, float)) or max_g < 0:
            return False, f"max_grid_window max_grid_kwh must be >= 0, got {max_g!r}"

    return True, ""


def deterministic_guardrail_validate(
    directives: List[Dict],
    num_notes: int,
) -> Tuple[bool, List[str]]:
    """
    Validate the full directive_interpretation array.

    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if not isinstance(directives, list):
        return False, ["directive_interpretation must be a list"]

    if len(directives) != num_notes:
        errors.append(
            f"Expected {num_notes} directive(s), got {len(directives)}"
        )
        return False, errors

    seen_indices = set()
    for i, entry in enumerate(directives):
        if not isinstance(entry, dict):
            errors.append(f"Entry {i} is not a dict")
            continue
        ok, msg = _validate_directive_entry(entry, num_notes)
        if not ok:
            errors.append(f"Entry {i} (note_index={entry.get('note_index', '?')}): {msg}")
        else:
            idx = entry["note_index"]
            if idx in seen_indices:
                errors.append(f"Duplicate note_index: {idx}")
            seen_indices.add(idx)

    if errors:
        logger.warning("Guardrail validation failed: %s", errors)
        return False, errors

    # Check indices cover exactly 0..num_notes-1
    if seen_indices != set(range(num_notes)):
        errors.append(f"note_index values must be 0..{num_notes-1}, got {sorted(seen_indices)}")
        return False, errors

    return True, []


def make_safe_fallback(num_notes: int) -> List[Dict]:
    """Return all-no_op fallback when retries are exhausted."""
    return [
        {
            "note_index": i,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Fallback: could not interpret note after maximum retries.",
        }
        for i in range(num_notes)
    ]
