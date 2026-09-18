#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path

TOL = 0.01

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000")
args = ap.parse_args()
case = json.loads((Path(__file__).parents[1] / "tests" / "data" / "sample01.json").read_text())

req = urllib.request.Request(
    args.base_url.rstrip("/") + "/optimize-energy",
    data=json.dumps(case["input"]).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as res:
    out = json.loads(res.read().decode())

expected = case["expected_output"]
assert out["scenario_id"] == expected["scenario_id"]
assert len(out["hourly_plan"]) == 24
assert abs(float(out["total_cost_bdt"]) - float(expected["total_cost_bdt"])) <= TOL

for got, exp in zip(out["directive_interpretation"], expected["directive_interpretation"]):
    assert got["note_index"] == exp["note_index"]
    assert got["applies"] == exp["applies"]
    assert got["directive_type"] == exp["directive_type"]
    assert got["structured_adjustment"] == exp["structured_adjustment"]

print("PASS SAMPLE-01")
