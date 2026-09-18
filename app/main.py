from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .interpreter import InterpretationError, interpret_notes, warmup_interpreter
from .models import OptimizeRequest, OptimizeResponse
from .optimizer import OptimizationError, optimize_schedule

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise BUP CSE Fest 2026", version="1.0.0", docs_url=None, redoc_url=None)


@app.on_event("startup")
def startup() -> None:
    warmup_interpreter()
    logger.info("GridWise service ready")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    is_json_decode = any(err.get("type") == "json_invalid" for err in exc.errors())
    status = 400 if is_json_decode else 422
    return JSONResponse(status_code=status, content={"detail": "invalid request"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(req: OptimizeRequest) -> OptimizeResponse:
    try:
        directives = interpret_notes(req)
        plan, total_grid, total_cost, peak = optimize_schedule(req, directives)
        active = [d.directive_type for d in directives if d.applies]
        if active:
            summary = (
                "Applied validated operator directives ("
                + ", ".join(active)
                + ") and minimized grid electricity cost while preserving all battery and energy constraints."
            )
        else:
            summary = "No operator note changed the energy constraints; minimized grid electricity cost under the base GridWise rules."
        return OptimizeResponse(
            scenario_id=req.scenario_id,
            directive_interpretation=directives,
            hourly_plan=plan,
            total_grid_kwh=total_grid,
            total_cost_bdt=total_cost,
            peak_grid_kwh=peak,
            plan_summary=summary,
        )
    except InterpretationError as exc:
        logger.warning("Controlled interpretation failure: %s", exc)
        raise HTTPException(status_code=500, detail="operator-note interpretation failed safely") from None
    except OptimizationError as exc:
        logger.warning("Controlled optimization failure: %s", exc)
        raise HTTPException(status_code=422, detail="scenario could not be optimized safely") from None
    except Exception:
        logger.exception("Unexpected controlled service error")
        raise HTTPException(status_code=500, detail="internal service error") from None
