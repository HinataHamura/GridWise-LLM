"""
app/optimizer.py — SciPy HiGHS LP optimizer for 24-hour energy scheduling.

Decision variables (120 total, h = 0..23):
  grid[h]       ∈ [0, grid_cap[h]]
  solar_used[h] ∈ [0, effective_solar[h]]
  charge[h]     ∈ [0, max_charge] or [0,0] if disallowed
  discharge[h]  ∈ [0, max_discharge] or [0,0] if disallowed
  E[h]          ∈ [min_reserve[h], capacity] ; E[23] pinned to initial_energy

Objective: minimize Σ tariff[h] * grid[h]

Equality constraints (48 rows):
  1. Energy balance:    grid[h] + solar_used[h] + discharge[h] - charge[h] = demand[h]
  2. Battery dynamics:  E[h] - E[h-1] - charge[h] + discharge[h] = 0
     (E[-1] = initial_energy_kwh as constant on rhs)

Post-processing: net = charge - discharge → battery_action (idle/charge/discharge)
"""
from __future__ import annotations

import math
import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from app.directives import HourConstraints
from app.schemas import BatteryInput, HourInput, HourlyPlanEntry

logger = logging.getLogger(__name__)

EPS = 1e-6  # threshold for charge/discharge net reconstruction


def _build_lp(
    hours_sorted: List[HourInput],
    battery: BatteryInput,
    constraints: HourConstraints,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build the LP arrays for scipy.optimize.linprog.

    Variable ordering (120 vars):
      0..23   → grid[0..23]
      24..47  → solar_used[0..23]
      48..71  → charge[0..23]
      72..95  → discharge[0..23]
      96..119 → E[0..23]

    Returns: (c, A_eq, b_eq, bounds)
    """
    N = 24
    NV = 5 * N  # 120 variables

    # Index helpers
    def ig(h): return h          # grid
    def is_(h): return N + h     # solar_used
    def ic(h): return 2*N + h    # charge
    def id_(h): return 3*N + h   # discharge
    def ie(h): return 4*N + h    # E

    # Objective: minimize Σ tariff[h] * grid[h]
    c_obj = np.zeros(NV)
    for h, hour in enumerate(hours_sorted):
        c_obj[ig(h)] = hour.tariff_bdt_per_kwh

    # Bounds
    cap = battery.capacity_kwh
    init_e = battery.initial_energy_kwh
    max_ch = battery.max_charge_kwh_per_hour
    max_disch = battery.max_discharge_kwh_per_hour

    bounds = []
    for h in range(N):
        bounds.append((0, constraints.grid_cap[h] if math.isfinite(constraints.grid_cap[h]) else None))  # grid
    for h in range(N):
        bounds.append((0, constraints.effective_solar[h]))  # solar_used
    for h in range(N):
        if constraints.charge_allowed[h]:
            bounds.append((0, max_ch))
        else:
            bounds.append((0, 0))  # no_charge_window
    for h in range(N):
        if constraints.discharge_allowed[h]:
            bounds.append((0, max_disch))
        else:
            bounds.append((0, 0))  # no_discharge_window
    for h in range(N):
        lo = constraints.min_reserve[h]
        hi = cap
        if h == N - 1:
            # E[23] pinned to initial_energy (end-of-day neutrality)
            lo = init_e
            hi = init_e
        bounds.append((lo, hi))

    # Equality constraints: 48 rows
    A_eq = np.zeros((2 * N, NV))
    b_eq = np.zeros(2 * N)

    for h, hour in enumerate(hours_sorted):
        # Row h: energy balance
        # grid[h] + solar_used[h] + discharge[h] - charge[h] = demand[h]
        A_eq[h, ig(h)] = 1.0
        A_eq[h, is_(h)] = 1.0
        A_eq[h, id_(h)] = 1.0
        A_eq[h, ic(h)] = -1.0
        b_eq[h] = hour.demand_kwh

        # Row N+h: battery dynamics
        # E[h] - charge[h] + discharge[h] = E[h-1]
        A_eq[N + h, ie(h)] = 1.0
        A_eq[N + h, ic(h)] = -1.0
        A_eq[N + h, id_(h)] = 1.0
        if h == 0:
            b_eq[N + h] = init_e  # E[-1] = initial_energy
        else:
            A_eq[N + h, ie(h - 1)] = -1.0
            b_eq[N + h] = 0.0

    return c_obj, A_eq, b_eq, bounds


def optimize(
    hours_sorted: List[HourInput],
    battery: BatteryInput,
    constraints: HourConstraints,
) -> Tuple[List[HourlyPlanEntry], float, float, float]:
    """
    Run the LP and return (hourly_plan, total_grid_kwh, total_cost_bdt, peak_grid_kwh).
    Raises RuntimeError if infeasible.
    """
    c_obj, A_eq, b_eq, bounds = _build_lp(hours_sorted, battery, constraints)

    res = linprog(
        c=c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": False},
    )

    if not res.success:
        raise RuntimeError(f"LP infeasible or unbounded: {res.message}")

    x = res.x
    N = 24

    def ig(h): return h
    def is_(h): return N + h
    def ic(h): return 2*N + h
    def id_(h): return 3*N + h
    def ie(h): return 4*N + h

    hourly_plan: List[HourlyPlanEntry] = []
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h, hour in enumerate(hours_sorted):
        grid_val = round(max(0.0, x[ig(h)]), 6)
        solar_val = round(max(0.0, x[is_(h)]), 6)
        charge_val = max(0.0, x[ic(h)])
        disch_val = max(0.0, x[id_(h)])
        e_val = round(x[ie(h)], 6)

        # Battery action reconstruction from net
        net = charge_val - disch_val
        if net > EPS:
            bat_action = "charge"
            bat_kwh = round(net, 6)
        elif net < -EPS:
            bat_action = "discharge"
            bat_kwh = round(-net, 6)
        else:
            bat_action = "idle"
            bat_kwh = 0.0

        entry = HourlyPlanEntry(
            hour=hour.hour,
            grid_kwh=grid_val,
            solar_used_kwh=solar_val,
            battery_action=bat_action,
            battery_kwh=bat_kwh,
            battery_energy_after_kwh=e_val,
        )
        hourly_plan.append(entry)

        total_grid += grid_val
        total_cost += grid_val * hour.tariff_bdt_per_kwh
        peak_grid = max(peak_grid, grid_val)

    total_grid = round(total_grid, 6)
    total_cost = round(total_cost, 6)
    peak_grid = round(peak_grid, 6)

    return hourly_plan, total_grid, total_cost, peak_grid
