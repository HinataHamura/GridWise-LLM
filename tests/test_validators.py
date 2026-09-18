"""
tests/test_validators.py — Unit tests for deterministic validators and final_validator.

These tests do NOT require any API keys — pure logic tests.
"""
import pytest

from app.validators import deterministic_guardrail_validate, make_safe_fallback


# ─── Guardrail validator tests ────────────────────────────────────────────────

class TestGuardrailValidate:
    def _valid_noop(self, idx=0):
        return {
            "note_index": idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Not relevant",
        }

    def _valid_solar(self, idx=0):
        return {
            "note_index": idx,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
            "explanation": "Solar panels cleaning",
        }

    def test_valid_single_noop(self):
        ok, errors = deterministic_guardrail_validate([self._valid_noop()], 1)
        assert ok, errors

    def test_valid_single_solar(self):
        ok, errors = deterministic_guardrail_validate([self._valid_solar()], 1)
        assert ok, errors

    def test_valid_two_notes(self):
        directives = [self._valid_solar(0), self._valid_noop(1)]
        ok, errors = deterministic_guardrail_validate(directives, 2)
        assert ok, errors

    def test_wrong_count(self):
        ok, errors = deterministic_guardrail_validate([self._valid_noop()], 2)
        assert not ok
        assert any("Expected 2" in e for e in errors)

    def test_noop_with_applies_true(self):
        d = self._valid_noop()
        d["applies"] = True
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_noop_with_adj(self):
        d = self._valid_noop()
        d["structured_adjustment"] = {"hours": [1]}
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_solar_with_applies_false(self):
        d = self._valid_solar()
        d["applies"] = False
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_solar_factor_too_high(self):
        d = self._valid_solar()
        d["structured_adjustment"]["factor"] = 1.5
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_solar_factor_zero(self):
        d = self._valid_solar()
        d["structured_adjustment"]["factor"] = 0.0
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_hours_unsorted(self):
        d = self._valid_solar()
        d["structured_adjustment"]["hours"] = [13, 12]  # descending — invalid
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_hours_duplicate(self):
        d = self._valid_solar()
        d["structured_adjustment"]["hours"] = [12, 12]
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_hours_out_of_range(self):
        d = self._valid_solar()
        d["structured_adjustment"]["hours"] = [24]
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok

    def test_duplicate_note_index(self):
        directives = [self._valid_solar(0), self._valid_noop(0)]  # both idx=0
        ok, errors = deterministic_guardrail_validate(directives, 2)
        assert not ok

    def test_valid_min_reserve(self):
        d = {
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0},
            "explanation": "Reserve",
        }
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert ok, errors

    def test_valid_no_charge(self):
        d = {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [2, 3, 4]},
            "explanation": "Maintenance",
        }
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert ok, errors

    def test_valid_max_grid(self):
        d = {
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [18, 19], "max_grid_kwh": 155.0},
            "explanation": "Feeder fault",
        }
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert ok, errors

    def test_invalid_directive_type(self):
        d = self._valid_solar()
        d["directive_type"] = "unknown_type"
        ok, errors = deterministic_guardrail_validate([d], 1)
        assert not ok


class TestSafeFallback:
    def test_makes_correct_count(self):
        fb = make_safe_fallback(3)
        assert len(fb) == 3
        for i, d in enumerate(fb):
            assert d["note_index"] == i
            assert d["applies"] is False
            assert d["directive_type"] == "no_op"
            assert d["structured_adjustment"] is None
