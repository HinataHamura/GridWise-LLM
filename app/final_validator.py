"""
app/final_validator.py — Independent replay validation of the optimizer output.

All comparisons use math.isclose(abs_tol=0.01) or abs(a-b)<=0.01 to handle
floating-point dust from the LP solver (never use strict == for floats).

Raises ValueError with a descriptive message if any check fails.
The optimizer's plan is NEVER sent to the client unless this passes.
"""
from __future__ import annotations

import math
import logging
from typing import List

from app.directives import HourConstraints
from app.schemas import BatteryInput, HourInput, HourlyPlanEntry

logger = logging.getLogger(__name__)

TOL = 0.01  # allowed absolute tolerance


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def validate_plan(
    hourly_plan: List[HourlyPlanEntry],
    hours_sorted: List[HourInput],
    battery: BatteryInput,
    constraints: HourConstraints,
    directives: List[dict],
    total_grid_kwh: float,
    total_cost_bdt: float,
    peak_grid_kwh: float,
) -> None:
    """
    Replay and verify the optimizer output against all constraints.
    Raises ValueError if any check fails.
    """
    N = 24
    assert len(hourly_plan) == N, f"hourly_plan must have 24 entries, got {len(hourly_plan)}"
    assert len(hours_sorted) == N

    demand = {h.hour: h.demand_kwh for h in hours_sorted}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours_sorted}

    recomputed_grid = 0.0
    recomputed_cost = 0.0
    recomputed_peak = 0.0
    prev_e = battery.initial_energy_kwh

    for entry in hourly_plan:
        h = entry.hour
        grid = entry.grid_kwh
        solar = entry.solar_used_kwh
        action = entry.battery_action
        bat_kwh = entry.battery_kwh
        e_after = entry.battery_energy_after_kwh

        # 1. solar_used ≤ effective_solar + TOL
        eff_solar = constraints.effective_solar[h]
        if solar > eff_solar + TOL:
            raise ValueError(
                f"Hour {h}: solar_used_kwh={solar:.4f} exceeds effective_solar={eff_solar:.4f}"
            )

        # 2. grid ≥ 0
        if grid < -TOL:
            raise ValueError(f"Hour {h}: grid_kwh={grid:.4f} is negative")

        # 3. grid ≤ grid_cap
        if grid > constraints.grid_cap[h] + TOL:
            raise ValueError(
                f"Hour {h}: grid_kwh={grid:.4f} exceeds cap={constraints.grid_cap[h]:.4f}"
            )

        # 4. battery_kwh ≥ 0; idle → battery_kwh == 0
        if bat_kwh < -TOL:
            raise ValueError(f"Hour {h}: battery_kwh={bat_kwh:.4f} is negative")
        if action == "idle" and bat_kwh > TOL:
            raise ValueError(f"Hour {h}: idle action but battery_kwh={bat_kwh:.4f} != 0")

        # 5. no_charge_window → action != charge
        if not constraints.charge_allowed[h] and action == "charge" and bat_kwh > TOL:
            raise ValueError(f"Hour {h}: charge prohibited (no_charge_window) but action=charge, kwh={bat_kwh:.4f}")

        # 6. no_discharge_window → action != discharge
        if not constraints.discharge_allowed[h] and action == "discharge" and bat_kwh > TOL:
            raise ValueError(f"Hour {h}: discharge prohibited (no_discharge_window) but action=discharge, kwh={bat_kwh:.4f}")

        # 7. Rate limits
        if action == "charge" and bat_kwh > battery.max_charge_kwh_per_hour + TOL:
            raise ValueError(
                f"Hour {h}: charge={bat_kwh:.4f} exceeds max_charge={battery.max_charge_kwh_per_hour}"
            )
        if action == "discharge" and bat_kwh > battery.max_discharge_kwh_per_hour + TOL:
            raise ValueError(
                f"Hour {h}: discharge={bat_kwh:.4f} exceeds max_discharge={battery.max_discharge_kwh_per_hour}"
            )

        # 8. Battery chain
        if action == "charge":
            expected_e = prev_e + bat_kwh
        elif action == "discharge":
            expected_e = prev_e - bat_kwh
        else:
            expected_e = prev_e

        if not _close(e_after, expected_e):
            raise ValueError(
                f"Hour {h}: battery_energy_after_kwh={e_after:.4f} != expected {expected_e:.4f}"
            )

        # 9. Battery bounds
        min_reserve = constraints.min_reserve[h]
        if e_after < min_reserve - TOL:
            raise ValueError(
                f"Hour {h}: battery_energy_after={e_after:.4f} < min_reserve={min_reserve:.4f}"
            )
        if e_after > battery.capacity_kwh + TOL:
            raise ValueError(
                f"Hour {h}: battery_energy_after={e_after:.4f} > capacity={battery.capacity_kwh}"
            )

        # 10. Energy balance: grid + solar + discharge - charge = demand
        charge_kwh = bat_kwh if action == "charge" else 0.0
        disch_kwh = bat_kwh if action == "discharge" else 0.0
        balance = grid + solar + disch_kwh - charge_kwh
        dem = demand[h]
        if not _close(balance, dem):
            raise ValueError(
                f"Hour {h}: energy balance {balance:.4f} != demand {dem:.4f}"
            )

        prev_e = e_after
        recomputed_grid += grid
        recomputed_cost += grid * tariff[h]
        recomputed_peak = max(recomputed_peak, grid)

    # 11. End-of-day neutrality
    if not _close(prev_e, battery.initial_energy_kwh):
        raise ValueError(
            f"End-of-day E[23]={prev_e:.4f} != initial_energy_kwh={battery.initial_energy_kwh:.4f}"
        )

    # 12. Summary field cross-check
    if not _close(recomputed_grid, total_grid_kwh):
        raise ValueError(
            f"total_grid_kwh={total_grid_kwh:.4f} != recomputed {recomputed_grid:.4f}"
        )
    if not _close(recomputed_cost, total_cost_bdt):
        raise ValueError(
            f"total_cost_bdt={total_cost_bdt:.4f} != recomputed {recomputed_cost:.4f}"
        )
    if not _close(recomputed_peak, peak_grid_kwh):
        raise ValueError(
            f"peak_grid_kwh={peak_grid_kwh:.4f} != recomputed {recomputed_peak:.4f}"
        )

    logger.info("Final validator: all checks passed ✓")
