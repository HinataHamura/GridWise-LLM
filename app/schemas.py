"""
app/schemas.py — All Pydantic models for GridWise LLM API (input & output).
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Input schemas ────────────────────────────────────────────────────────────

class HourInput(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class BatteryInput(BaseModel):
    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)


class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourInput] = Field(..., min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("hours")
    @classmethod
    def validate_hours_complete(cls, v: List[HourInput]) -> List[HourInput]:
        hour_set = {h.hour for h in v}
        if hour_set != set(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0-23")
        return sorted(v, key=lambda x: x.hour)

    @field_validator("operator_notes")
    @classmethod
    def validate_notes_nonempty(cls, v: List[str]) -> List[str]:
        for note in v:
            if not note.strip():
                raise ValueError("operator_notes must be non-empty strings")
        return v


# ─── Directive output schemas ─────────────────────────────────────────────────

class SolarReductionAdjustment(BaseModel):
    hours: List[int]
    factor: float = Field(..., gt=0, le=1)


class MinBatteryReserveAdjustment(BaseModel):
    hours: List[int]
    minimum_energy_kwh: float = Field(..., ge=0)


class NoChargeWindowAdjustment(BaseModel):
    hours: List[int]


class NoDischargeWindowAdjustment(BaseModel):
    hours: List[int]


class MaxGridWindowAdjustment(BaseModel):
    hours: List[int]
    max_grid_kwh: float = Field(..., ge=0)


# Union type for structured_adjustment — stored as dict internally
StructuredAdjustment = dict  # validated by guardrail


class DirectiveEntry(BaseModel):
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    ]
    structured_adjustment: Optional[dict] = None
    explanation: str


# ─── Output schemas ───────────────────────────────────────────────────────────

class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveEntry]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
