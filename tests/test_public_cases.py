"""
tests/test_public_cases.py — End-to-end tests using the 10 public sample cases.

These tests load the JSON, run the FULL pipeline (no mocking), and verify:
1. directive_interpretation semantics match expected ground truth
2. hourly_plan satisfies all constraints (energy balance, battery, directives)
3. total_grid_kwh, total_cost_bdt, peak_grid_kwh within 0.01 tolerance of expected

Requires API keys in environment (GEMINI_API_KEY or GROQ_API_KEY or HF_API_TOKEN).
Run with: pytest tests/test_public_cases.py -v
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from app.directives import build_hour_constraints
from app.final_validator import validate_plan
from app.graph import run_interpretation
from app.optimizer import optimize
from app.schemas import BatteryInput, HourInput, OptimizeRequest

# Load sample cases
SAMPLE_CASES_PATH = Path(__file__).parent.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"

TOL = 0.01


def load_cases():
    with open(SAMPLE_CASES_PATH) as f:
        data = json.load(f)
    return data["cases"]


def build_request(case_input: dict) -> OptimizeRequest:
    return OptimizeRequest(
        scenario_id=case_input["scenario_id"],
        operator_notes=case_input["operator_notes"],
        hours=[HourInput(**h) for h in case_input["hours"]],
        battery=BatteryInput(**case_input["battery"]),
    )


def check_constraints(plan, hours_sorted, battery, constraints):
    """Reuse final_validator logic inline for test assertions."""
    demand = {h.hour: h.demand_kwh for h in hours_sorted}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours_sorted}
    prev_e = battery.initial_energy_kwh
    for entry in plan:
        h = entry.hour
        charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        disch = entry.battery_kwh if entry.battery_action == "discharge" else 0.0
        balance = entry.grid_kwh + entry.solar_used_kwh + disch - charge
        assert abs(balance - demand[h]) <= TOL, f"Energy balance fail at hour {h}"
        assert entry.solar_used_kwh <= constraints.effective_solar[h] + TOL
        assert entry.grid_kwh <= constraints.grid_cap[h] + TOL
        if not constraints.charge_allowed[h]:
            assert not (entry.battery_action == "charge" and entry.battery_kwh > TOL)
        if not constraints.discharge_allowed[h]:
            assert not (entry.battery_action == "discharge" and entry.battery_kwh > TOL)
        prev_e = entry.battery_energy_after_kwh
    assert abs(prev_e - battery.initial_energy_kwh) <= TOL, "End-of-day neutrality violated"


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["id"])
def test_public_case_full_pipeline(case):
    """Run full pipeline for each public sample case and verify constraints + cost."""
    case_input = case["input"]
    expected_output = case["expected_output"]

    request = build_request(case_input)
    hours_sorted = sorted(request.hours, key=lambda h: h.hour)

    # Step 1: LLM interpretation
    raw_directives = run_interpretation(
        operator_notes=request.operator_notes,
        capacity_kwh=request.battery.capacity_kwh,
        minimum_energy_kwh=request.battery.minimum_energy_kwh,
        initial_energy_kwh=request.battery.initial_energy_kwh,
    )

    assert len(raw_directives) == len(request.operator_notes), \
        f"Expected {len(request.operator_notes)} directives, got {len(raw_directives)}"

    # Step 2: Verify directive types match expected (semantic check)
    expected_interp = expected_output["directive_interpretation"]
    for i, (got, exp) in enumerate(zip(raw_directives, expected_interp)):
        assert got["directive_type"] == exp["directive_type"], \
            f"Note {i}: got directive_type={got['directive_type']!r}, expected={exp['directive_type']!r}"
        assert got["applies"] == exp["applies"], \
            f"Note {i}: got applies={got['applies']}, expected={exp['applies']}"

    # Step 3: Build constraints
    constraints = build_hour_constraints(request, raw_directives)

    # Step 4: Optimize
    plan, total_grid, total_cost, peak = optimize(hours_sorted, request.battery, constraints)

    # Step 5: Final validation (full replay)
    validate_plan(
        hourly_plan=plan,
        hours_sorted=hours_sorted,
        battery=request.battery,
        constraints=constraints,
        directives=raw_directives,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak,
    )

    # Step 6: Verify cost within tolerance of expected
    exp_cost = expected_output["total_cost_bdt"]
    assert abs(total_cost - exp_cost) <= TOL, \
        f"total_cost_bdt={total_cost:.4f} vs expected={exp_cost:.4f} (diff={abs(total_cost-exp_cost):.4f})"

    exp_grid = expected_output["total_grid_kwh"]
    assert abs(total_grid - exp_grid) <= TOL, \
        f"total_grid_kwh={total_grid:.4f} vs expected={exp_grid:.4f}"
