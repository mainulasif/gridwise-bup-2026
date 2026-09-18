import json
from pathlib import Path

from app.interpreter import RawLLMResult, _guardrail_and_normalize
from app.models import DirectiveInterpretation, OptimizeRequest
from app.optimizer import optimize_schedule

PACK = json.loads((Path(__file__).parent / "data" / "sample01.json").read_text())


def test_public_sample_optimizer_cost():
    case = PACK
    req = OptimizeRequest.model_validate(case["input"])
    directives = [
        DirectiveInterpretation.model_validate(x)
        for x in case["expected_output"]["directive_interpretation"]
    ]
    _, total_grid, total_cost, _ = optimize_schedule(req, directives)
    expected = case["expected_output"]
    assert abs(total_grid - expected["total_grid_kwh"]) <= 0.01
    assert abs(total_cost - expected["total_cost_bdt"]) <= 0.01


def test_public_sample_normalization():
    case = PACK
    req = OptimizeRequest.model_validate(case["input"])
    for idx, expected in enumerate(case["expected_output"]["directive_interpretation"]):
        raw = RawLLMResult(
            directive_type=expected["directive_type"],
            applies=None,
            hours=None,
        )
        got = _guardrail_and_normalize(idx, req.operator_notes[idx], raw, req)
        assert got.directive_type == expected["directive_type"]
        assert got.applies == expected["applies"]
        assert got.structured_adjustment == expected["structured_adjustment"]
