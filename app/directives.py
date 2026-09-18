"""
app/directives.py — Translate validated directive_interpretation into per-hour arrays.

Handles overlapping directives from multiple notes (corner case):
  - solar_reduction:           effective_solar[h] *= factor per directive (multiply factors)
  - minimum_battery_reserve:   min_reserve[h] = max(base, res1, res2, ...)
  - max_grid_window:           grid_cap[h] = min(cap1, cap2, ...)
  - no_charge_window:          charge_allowed[h] = False
  - no_discharge_window:       discharge_allowed[h] = False
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from app.schemas import BatteryInput, HourInput, OptimizeRequest


@dataclass
class HourConstraints:
    """Per-hour constraint arrays derived from directives + base battery limits."""
    effective_solar: List[float] = field(default_factory=lambda: [0.0] * 24)
    min_reserve: List[float] = field(default_factory=lambda: [0.0] * 24)
    charge_allowed: List[bool] = field(default_factory=lambda: [True] * 24)
    discharge_allowed: List[bool] = field(default_factory=lambda: [True] * 24)
    grid_cap: List[float] = field(default_factory=lambda: [math.inf] * 24)


def build_hour_constraints(
    request: OptimizeRequest,
    directives: List[dict],
) -> HourConstraints:
    """
    Build per-hour constraint arrays from request data and validated directives.

    All 24 hours addressed; base values from raw solar and battery specs.
    Multiple directives on the same hour: most restrictive wins.
    """
    battery: BatteryInput = request.battery
    hours_sorted = sorted(request.hours, key=lambda h: h.hour)

    # Initialize with base values
    c = HourConstraints()
    for h_obj in hours_sorted:
        h = h_obj.hour
        c.effective_solar[h] = h_obj.solar_kwh
        c.min_reserve[h] = battery.minimum_energy_kwh
        c.charge_allowed[h] = True
        c.discharge_allowed[h] = True
        c.grid_cap[h] = math.inf

    # Apply each directive
    for entry in directives:
        if not entry.get("applies", False):
            continue  # no_op or applies=False

        dtype = entry["directive_type"]
        adj = entry.get("structured_adjustment") or {}
        hours: List[int] = adj.get("hours", [])

        if dtype == "solar_reduction":
            factor = float(adj.get("factor", 1.0))
            for h in hours:
                c.effective_solar[h] *= factor  # multiply if multiple reductions

        elif dtype == "minimum_battery_reserve":
            min_e = float(adj.get("minimum_energy_kwh", 0.0))
            for h in hours:
                # Most restrictive = maximum reserve
                c.min_reserve[h] = max(c.min_reserve[h], min_e)

        elif dtype == "no_charge_window":
            for h in hours:
                c.charge_allowed[h] = False

        elif dtype == "no_discharge_window":
            for h in hours:
                c.discharge_allowed[h] = False

        elif dtype == "max_grid_window":
            max_g = float(adj.get("max_grid_kwh", math.inf))
            for h in hours:
                # Most restrictive = minimum cap
                c.grid_cap[h] = min(c.grid_cap[h], max_g)

    return c
