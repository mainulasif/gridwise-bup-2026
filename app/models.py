from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]
BatteryAction = Literal["charge", "discharge", "idle"]


def _finite_nonnegative(v: float, name: str) -> float:
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(v)


class HourInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hour: int = Field(ge=0, le=23)
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float

    @field_validator("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")
    @classmethod
    def values_are_valid(cls, v: float, info):
        return _finite_nonnegative(v, info.field_name)


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float

    @field_validator(
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
    )
    @classmethod
    def values_are_valid(cls, v: float, info):
        return _finite_nonnegative(v, info.field_name)

    @model_validator(mode="after")
    def bounds_are_consistent(self):
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if not (self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh):
            raise ValueError("initial_energy_kwh must be within battery bounds")
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str = Field(min_length=1, max_length=200)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("scenario_id")
    @classmethod
    def scenario_not_blank(cls, v: str):
        if not v.strip():
            raise ValueError("scenario_id must not be blank")
        return v

    @field_validator("operator_notes")
    @classmethod
    def notes_not_blank(cls, notes: list[str]):
        if any(not isinstance(n, str) or not n.strip() for n in notes):
            raise ValueError("operator_notes must contain non-empty strings")
        return [n.strip() for n in notes]

    @model_validator(mode="after")
    def hours_cover_day(self):
        hs = sorted(h.hour for h in self.hours)
        if hs != list(range(24)):
            raise ValueError("hours must contain exactly one entry for every hour 0 through 23")
        return self

    def hours_in_order(self) -> list[HourInput]:
        return sorted(self.hours, key=lambda x: x.hour)


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict]
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
