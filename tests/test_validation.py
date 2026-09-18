import pytest
from pydantic import ValidationError

from app.models import OptimizeRequest


def _base():
    return {
        "scenario_id": "T",
        "operator_notes": ["The cafeteria menu changes tomorrow."],
        "hours": [
            {"hour": h, "demand_kwh": 10, "solar_kwh": 0, "tariff_bdt_per_kwh": 5}
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": 100,
            "initial_energy_kwh": 50,
            "minimum_energy_kwh": 20,
            "max_charge_kwh_per_hour": 20,
            "max_discharge_kwh_per_hour": 20,
        },
    }


def test_duplicate_hour_rejected():
    x = _base()
    x["hours"][23]["hour"] = 22
    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(x)


def test_negative_energy_rejected():
    x = _base()
    x["hours"][3]["demand_kwh"] = -1
    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(x)


def test_blank_note_rejected():
    x = _base()
    x["operator_notes"] = ["  "]
    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(x)
