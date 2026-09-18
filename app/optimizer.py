from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .models import DirectiveInterpretation, HourlyPlanEntry, OptimizeRequest

TOL = 1e-7


class OptimizationError(RuntimeError):
    pass


@dataclass
class AppliedConstraints:
    effective_solar: list[float]
    minimum_energy: list[float]
    charge_allowed: list[bool]
    discharge_allowed: list[bool]
    max_grid: list[float | None]


def apply_directives(req: OptimizeRequest, directives: list[DirectiveInterpretation]) -> AppliedConstraints:
    hours = req.hours_in_order()
    effective_solar = [float(h.solar_kwh) for h in hours]
    minimum_energy = [float(req.battery.minimum_energy_kwh)] * 24
    charge_allowed = [True] * 24
    discharge_allowed = [True] * 24
    max_grid: list[float | None] = [None] * 24

    for d in directives:
        if not d.applies or d.directive_type == "no_op":
            continue
        a = d.structured_adjustment or {}
        hs = a.get("hours", [])
        if d.directive_type == "solar_reduction":
            factor = float(a["factor"])
            for h in hs:
                effective_solar[h] = min(effective_solar[h], hours[h].solar_kwh * factor)
        elif d.directive_type == "minimum_battery_reserve":
            reserve = float(a["minimum_energy_kwh"])
            for h in hs:
                minimum_energy[h] = max(minimum_energy[h], reserve)
        elif d.directive_type == "no_charge_window":
            for h in hs:
                charge_allowed[h] = False
        elif d.directive_type == "no_discharge_window":
            for h in hs:
                discharge_allowed[h] = False
        elif d.directive_type == "max_grid_window":
            cap = float(a["max_grid_kwh"])
            for h in hs:
                max_grid[h] = cap if max_grid[h] is None else min(max_grid[h], cap)

    return AppliedConstraints(
        effective_solar=effective_solar,
        minimum_energy=minimum_energy,
        charge_allowed=charge_allowed,
        discharge_allowed=discharge_allowed,
        max_grid=max_grid,
    )


def optimize_schedule(req: OptimizeRequest, directives: list[DirectiveInterpretation]):
    hs = req.hours_in_order()
    cst = apply_directives(req, directives)
    n = 24
    G, S, B, E = 0, n, 2 * n, 3 * n
    size = 4 * n

    obj = np.zeros(size)
    for h in range(n):
        obj[G + h] = hs[h].tariff_bdt_per_kwh

    bounds: list[tuple[float | None, float | None]] = []
    for h in range(n):
        bounds.append((0.0, cst.max_grid[h]))
    for h in range(n):
        bounds.append((0.0, max(0.0, cst.effective_solar[h])))
    for h in range(n):
        lo = -req.battery.max_charge_kwh_per_hour if cst.charge_allowed[h] else 0.0
        hi = req.battery.max_discharge_kwh_per_hour if cst.discharge_allowed[h] else 0.0
        bounds.append((lo, hi))
    for h in range(n):
        bounds.append((cst.minimum_energy[h], req.battery.capacity_kwh))

    eq_rows = []
    eq_rhs = []

    for h in range(n):
        row = np.zeros(size)
        row[G + h] = 1.0
        row[S + h] = 1.0
        row[B + h] = 1.0
        eq_rows.append(row)
        eq_rhs.append(hs[h].demand_kwh)

    for h in range(n):
        row = np.zeros(size)
        row[E + h] = 1.0
        row[B + h] = 1.0
        if h == 0:
            rhs = req.battery.initial_energy_kwh
        else:
            row[E + h - 1] = -1.0
            rhs = 0.0
        eq_rows.append(row)
        eq_rhs.append(rhs)

    row = np.zeros(size)
    row[E + 23] = 1.0
    eq_rows.append(row)
    eq_rhs.append(req.battery.initial_energy_kwh)

    result = linprog(
        c=obj,
        A_eq=np.array(eq_rows),
        b_eq=np.array(eq_rhs),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise OptimizationError(f"optimization failed: {result.message}")

    x = result.x
    plan: list[HourlyPlanEntry] = []
    for h in range(n):
        grid = max(0.0, float(x[G + h]))
        solar = max(0.0, float(x[S + h]))
        flow = float(x[B + h])
        energy = float(x[E + h])
        if flow > TOL:
            action, batt = "discharge", flow
        elif flow < -TOL:
            action, batt = "charge", -flow
        else:
            action, batt = "idle", 0.0
        plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=_clean(grid),
            solar_used_kwh=_clean(solar),
            battery_action=action,
            battery_kwh=_clean(batt),
            battery_energy_after_kwh=_clean(energy),
        ))

    validate_replay(req, directives, plan)
    total_grid = _clean(sum(p.grid_kwh for p in plan))
    total_cost = _clean(sum(plan[h].grid_kwh * hs[h].tariff_bdt_per_kwh for h in range(24)))
    peak_grid = _clean(max(p.grid_kwh for p in plan))
    return plan, total_grid, total_cost, peak_grid


def validate_replay(
    req: OptimizeRequest,
    directives: list[DirectiveInterpretation],
    plan: list[HourlyPlanEntry],
    tol: float = 0.009,
) -> None:
    if len(plan) != 24 or [p.hour for p in plan] != list(range(24)):
        raise OptimizationError("replay failed: hourly_plan must contain hours 0..23 in order")

    hs = req.hours_in_order()
    cst = apply_directives(req, directives)
    before = req.battery.initial_energy_kwh

    for h, p in enumerate(plan):
        vals = [p.grid_kwh, p.solar_used_kwh, p.battery_kwh, p.battery_energy_after_kwh]
        if not all(math.isfinite(v) and v >= -tol for v in vals):
            raise OptimizationError(f"replay failed: invalid numeric value at hour {h}")
        if p.solar_used_kwh > cst.effective_solar[h] + tol:
            raise OptimizationError(f"replay failed: solar overuse at hour {h}")
        if cst.max_grid[h] is not None and p.grid_kwh > cst.max_grid[h] + tol:
            raise OptimizationError(f"replay failed: grid cap exceeded at hour {h}")

        if p.battery_action == "charge":
            if not cst.charge_allowed[h] or p.battery_kwh > req.battery.max_charge_kwh_per_hour + tol:
                raise OptimizationError(f"replay failed: invalid charge at hour {h}")
            after = before + p.battery_kwh
            discharge, charge = 0.0, p.battery_kwh
        elif p.battery_action == "discharge":
            if not cst.discharge_allowed[h] or p.battery_kwh > req.battery.max_discharge_kwh_per_hour + tol:
                raise OptimizationError(f"replay failed: invalid discharge at hour {h}")
            after = before - p.battery_kwh
            discharge, charge = p.battery_kwh, 0.0
        else:
            if abs(p.battery_kwh) > tol:
                raise OptimizationError(f"replay failed: idle battery_kwh must be zero at hour {h}")
            after = before
            discharge = charge = 0.0

        if abs(after - p.battery_energy_after_kwh) > tol:
            raise OptimizationError(f"replay failed: battery transition mismatch at hour {h}")
        if p.battery_energy_after_kwh < cst.minimum_energy[h] - tol:
            raise OptimizationError(f"replay failed: reserve violation at hour {h}")
        if p.battery_energy_after_kwh > req.battery.capacity_kwh + tol:
            raise OptimizationError(f"replay failed: capacity violation at hour {h}")

        lhs = p.grid_kwh + p.solar_used_kwh + discharge
        rhs = hs[h].demand_kwh + charge
        if abs(lhs - rhs) > tol:
            raise OptimizationError(f"replay failed: energy balance mismatch at hour {h}")
        before = p.battery_energy_after_kwh

    if abs(before - req.battery.initial_energy_kwh) > tol:
        raise OptimizationError("replay failed: end-of-day battery neutrality violated")


def _clean(v: float) -> float:
    if abs(v) < 5e-8:
        return 0.0
    r = round(float(v), 6)
    return int(r) if abs(r - round(r)) < 1e-9 else r
