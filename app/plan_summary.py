"""
app/plan_summary.py — Template-based plan summary (no LLM call).

Per PDF Sec 02: plan_summary is a human-readable text field;
using LLM here does NOT satisfy the LLM interpretation requirement,
so we save API quota and use a simple template instead.
"""
from __future__ import annotations

from typing import List


DIRECTIVE_LABELS = {
    "solar_reduction": "solar reduction",
    "minimum_battery_reserve": "minimum battery reserve",
    "no_charge_window": "no-charge window",
    "no_discharge_window": "no-discharge window",
    "max_grid_window": "grid import cap",
    "no_op": None,
}


def build_plan_summary(
    directives: List[dict],
    total_cost_bdt: float,
    total_grid_kwh: float,
) -> str:
    """Build a 1-2 sentence plan summary from applied directives and cost."""
    applied_labels = []
    for d in directives:
        if d.get("applies", False) and d.get("directive_type") != "no_op":
            label = DIRECTIVE_LABELS.get(d["directive_type"])
            if label and label not in applied_labels:
                applied_labels.append(label)

    if not applied_labels:
        directive_part = "No active energy directives were applied."
    elif len(applied_labels) == 1:
        directive_part = f"Applied {applied_labels[0]} directive as instructed."
    else:
        labels_str = ", ".join(applied_labels[:-1]) + f", and {applied_labels[-1]}"
        directive_part = f"Applied {labels_str} directives as instructed."

    cost_part = (
        f"Optimized 24-hour schedule uses {total_grid_kwh:.2f} kWh from the grid "
        f"at a total cost of {total_cost_bdt:.2f} BDT."
    )

    return f"{directive_part} {cost_part}"
