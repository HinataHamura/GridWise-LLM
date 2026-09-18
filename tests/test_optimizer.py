"""
tests/test_optimizer.py — Unit tests for directives.py and optimizer.py.

Tests the LP optimizer in isolation (no LLM calls).
Validates that:
- Energy balance holds every hour
- Battery constraints are respected
- End-of-day neutrality is enforced
- Directives (no_charge, no_discharge, max_grid, solar_reduction, reserve) are applied
"""
import math
import pytest

from app.directives import HourConstraints, build_hour_constraints
from app.optimizer import optimize
from app.schemas import BatteryInput, HourInput, OptimizeRequest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_battery(
    capacity=200.0, initial=100.0, minimum=0.0, max_ch=50.0, max_disch=50.0
):
    return BatteryInput(
        capacity_kwh=capacity,
        initial_energy_kwh=initial,
        minimum_energy_kwh=minimum,
        max_charge_kwh_per_hour=max_ch,
        max_discharge_kwh_per_hour=max_disch,
    )


def make_hours(demand=100.0, solar=0.0, tariff=8.0):
    """Create 24 identical hours."""
    return [
        HourInput(hour=h, demand_kwh=demand, solar_kwh=solar, tariff_bdt_per_kwh=tariff)
        for h in range(24)
    ]


def make_request(battery, hours, notes=None):
    return OptimizeRequest(
        scenario_id="TEST",
        operator_notes=notes or ["No relevant note."],
        hours=hours,
        battery=battery,
    )


def make_constraints_no_directive(battery, hours):
    """Default constraints — no directives applied."""
    c = HourConstraints()
    for h_obj in hours:
        h = h_obj.hour
        c.effective_solar[h] = h_obj.solar_kwh
        c.min_reserve[h] = battery.minimum_energy_kwh
    return c


TOL = 0.01


# ─── Basic optimizer tests ─────────────────────────────────────────────────────

class TestOptimizerBasic:
    def test_energy_balance_holds(self):
        battery = make_battery()
        hours = make_hours(demand=100, solar=0)
        constraints = make_constraints_no_directive(battery, hours)

        plan, total_grid, total_cost, peak = optimize(hours, battery, constraints)

        for entry in plan:
            h_obj = hours[entry.hour]
            charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
            disch = entry.battery_kwh if entry.battery_action == "discharge" else 0.0
            balance = entry.grid_kwh + entry.solar_used_kwh + disch - charge
            assert abs(balance - h_obj.demand_kwh) <= TOL, f"Balance fail at hour {entry.hour}"

    def test_end_of_day_neutrality(self):
        battery = make_battery(initial=100.0)
        hours = make_hours()
        constraints = make_constraints_no_directive(battery, hours)
        plan, *_ = optimize(hours, battery, constraints)
        final_e = plan[-1].battery_energy_after_kwh
        assert abs(final_e - battery.initial_energy_kwh) <= TOL

    def test_total_grid_correct(self):
        battery = make_battery()
        hours = make_hours(demand=100)
        constraints = make_constraints_no_directive(battery, hours)
        plan, total_grid, total_cost, peak = optimize(hours, battery, constraints)

        recomputed = sum(e.grid_kwh for e in plan)
        assert abs(recomputed - total_grid) <= TOL

    def test_total_cost_correct(self):
        battery = make_battery()
        hours = make_hours(demand=100, tariff=8)
        constraints = make_constraints_no_directive(battery, hours)
        plan, total_grid, total_cost, peak = optimize(hours, battery, constraints)

        recomputed_cost = sum(e.grid_kwh * hours[e.hour].tariff_bdt_per_kwh for e in plan)
        assert abs(recomputed_cost - total_cost) <= TOL

    def test_solar_reduces_grid(self):
        """With free solar, grid usage should decrease."""
        battery = make_battery()
        hours_no_solar = make_hours(demand=100, solar=0)
        hours_with_solar = make_hours(demand=100, solar=50)

        c_no = make_constraints_no_directive(battery, hours_no_solar)
        c_with = make_constraints_no_directive(battery, hours_with_solar)

        _, grid_no_solar, _, _ = optimize(hours_no_solar, battery, c_no)
        _, grid_with_solar, _, _ = optimize(hours_with_solar, battery, c_with)

        assert grid_with_solar < grid_no_solar - TOL

    def test_idle_has_zero_battery_kwh(self):
        battery = make_battery()
        hours = make_hours(demand=100)
        constraints = make_constraints_no_directive(battery, hours)
        plan, *_ = optimize(hours, battery, constraints)
        for entry in plan:
            if entry.battery_action == "idle":
                assert entry.battery_kwh <= TOL


# ─── Directive constraint tests ───────────────────────────────────────────────

class TestDirectiveConstraints:
    def test_no_charge_window_respected(self):
        battery = make_battery()
        hours = make_hours(demand=100)
        constraints = make_constraints_no_directive(battery, hours)
        # Prohibit charging in hours 2, 3, 4
        for h in [2, 3, 4]:
            constraints.charge_allowed[h] = False

        plan, *_ = optimize(hours, battery, constraints)
        for entry in plan:
            if entry.hour in [2, 3, 4]:
                assert entry.battery_action != "charge" or entry.battery_kwh <= TOL

    def test_no_discharge_window_respected(self):
        battery = make_battery()
        hours = make_hours(demand=100)
        constraints = make_constraints_no_directive(battery, hours)
        for h in [18, 19]:
            constraints.discharge_allowed[h] = False

        plan, *_ = optimize(hours, battery, constraints)
        for entry in plan:
            if entry.hour in [18, 19]:
                assert entry.battery_action != "discharge" or entry.battery_kwh <= TOL

    def test_max_grid_cap_respected(self):
        battery = make_battery(capacity=200, initial=100, max_disch=50)
        hours = make_hours(demand=200, solar=0)  # High demand
        constraints = make_constraints_no_directive(battery, hours)
        cap = 155.0
        for h in [18, 19, 20]:
            constraints.grid_cap[h] = cap

        plan, *_ = optimize(hours, battery, constraints)
        for entry in plan:
            if entry.hour in [18, 19, 20]:
                assert entry.grid_kwh <= cap + TOL

    def test_min_reserve_respected(self):
        battery = make_battery(capacity=200, initial=200, minimum=0)
        hours = make_hours(demand=100)
        constraints = make_constraints_no_directive(battery, hours)
        reserve = 100.0
        for h in [18, 19, 20]:
            constraints.min_reserve[h] = reserve

        plan, *_ = optimize(hours, battery, constraints)
        for entry in plan:
            if entry.hour in [18, 19, 20]:
                assert entry.battery_energy_after_kwh >= reserve - TOL

    def test_solar_reduction_applied(self):
        battery = make_battery()
        hours = make_hours(demand=100, solar=100)
        constraints = make_constraints_no_directive(battery, hours)
        # 80% reduction for hours 11-13 → factor 0.2
        for h in [11, 12, 13]:
            constraints.effective_solar[h] = hours[h].solar_kwh * 0.2  # 20

        plan, *_ = optimize(hours, battery, constraints)
        for entry in plan:
            if entry.hour in [11, 12, 13]:
                assert entry.solar_used_kwh <= 20 + TOL


# ─── Directives.py build_hour_constraints tests ───────────────────────────────

class TestBuildHourConstraints:
    def _base_request(self):
        battery = make_battery(capacity=200, minimum=10)
        hours = make_hours(solar=50)
        return OptimizeRequest(
            scenario_id="T",
            operator_notes=["note"],
            hours=hours,
            battery=battery,
        )

    def test_solar_reduction_multiplies_factor(self):
        req = self._base_request()
        directives = [{
            "note_index": 0, "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
            "explanation": "",
        }]
        c = build_hour_constraints(req, directives)
        assert abs(c.effective_solar[12] - 50 * 0.25) <= 1e-9
        assert abs(c.effective_solar[0] - 50.0) <= 1e-9  # unchanged

    def test_overlapping_solar_reductions_multiply(self):
        """Two solar_reduction directives on same hour → factors multiply."""
        req = self._base_request()
        directives = [
            {
                "note_index": 0, "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [12], "factor": 0.5},
                "explanation": "",
            },
            {
                "note_index": 1, "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [12], "factor": 0.5},
                "explanation": "",
            },
        ]
        # Need 2 notes in request
        req.operator_notes = ["note1", "note2"]
        c = build_hour_constraints(req, directives)
        assert abs(c.effective_solar[12] - 50 * 0.5 * 0.5) <= 1e-9

    def test_overlapping_reserves_take_max(self):
        req = self._base_request()
        directives = [
            {
                "note_index": 0, "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {"hours": [18], "minimum_energy_kwh": 80.0},
                "explanation": "",
            },
        ]
        c = build_hour_constraints(req, directives)
        assert c.min_reserve[18] == max(10.0, 80.0)  # base=10, directive=80

    def test_overlapping_grid_caps_take_min(self):
        req = self._base_request()
        req.operator_notes = ["note1", "note2"]
        directives = [
            {
                "note_index": 0, "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": [18], "max_grid_kwh": 200.0},
                "explanation": "",
            },
            {
                "note_index": 1, "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": [18], "max_grid_kwh": 155.0},
                "explanation": "",
            },
        ]
        c = build_hour_constraints(req, directives)
        assert c.grid_cap[18] == 155.0  # min of 200 and 155
