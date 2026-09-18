"""
app/main.py — FastAPI application entry point for GridWise LLM.

Endpoints:
  GET  /health          → {"status": "ok"}  (no LLM dependency, fast cold-start check)
  POST /optimize-energy → full pipeline: interpret → optimize → validate → respond

Global exception handler ensures no stack traces or secrets leak to clients.
"""
from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.directives import build_hour_constraints
from app.final_validator import validate_plan
from app.graph import run_interpretation
from app.optimizer import optimize
from app.plan_summary import build_plan_summary
from app.schemas import DirectiveEntry, OptimizeRequest, OptimizeResponse

# Load .env for local development (Vercel provides env vars directly)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up few-shot embeddings on startup."""
    logger.info("GridWise LLM API starting up...")
    from app.fewshot import _warm_up_embeddings
    _warm_up_embeddings()
    logger.info("Startup complete")
    yield
    logger.info("GridWise LLM API shutting down")


app = FastAPI(
    title="GridWise LLM API",
    description="LLM-assisted 24-hour energy optimization — BUP CSE Fest 2026",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please retry."},
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: OptimizeRequest):
    logger.info("Received request for scenario_id=%s", request.scenario_id)

    # Step 1: Sort hours (validator already ensures exactly 24 unique hours)
    hours_sorted = sorted(request.hours, key=lambda h: h.hour)

    # Step 2: LangGraph interpretation pipeline
    logger.info("Running LLM interpretation for %d note(s)...", len(request.operator_notes))
    raw_directives: list = run_interpretation(
        operator_notes=request.operator_notes,
        capacity_kwh=request.battery.capacity_kwh,
        minimum_energy_kwh=request.battery.minimum_energy_kwh,
        initial_energy_kwh=request.battery.initial_energy_kwh,
    )

    # Convert to Pydantic DirectiveEntry objects
    directive_entries = [DirectiveEntry(**d) for d in raw_directives]

    # Step 3: Build per-hour constraint arrays
    hour_constraints = build_hour_constraints(request, raw_directives)

    # Step 4: Run LP optimizer
    logger.info("Running LP optimizer...")
    hourly_plan, total_grid_kwh, total_cost_bdt, peak_grid_kwh = optimize(
        hours_sorted=hours_sorted,
        battery=request.battery,
        constraints=hour_constraints,
    )

    # Step 5: Final validation (replay)
    logger.info("Running final validator...")
    try:
        validate_plan(
            hourly_plan=hourly_plan,
            hours_sorted=hours_sorted,
            battery=request.battery,
            constraints=hour_constraints,
            directives=raw_directives,
            total_grid_kwh=total_grid_kwh,
            total_cost_bdt=total_cost_bdt,
            peak_grid_kwh=peak_grid_kwh,
        )
    except ValueError as e:
        logger.error("Final validator failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Plan validation failed: {e}"},
        )

    # Step 6: Build plan summary
    plan_summary = build_plan_summary(raw_directives, total_cost_bdt, total_grid_kwh)

    # Step 7: Assemble and return response
    response = OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directive_entries,
        hourly_plan=hourly_plan,
        total_grid_kwh=total_grid_kwh,
        total_cost_bdt=total_cost_bdt,
        peak_grid_kwh=peak_grid_kwh,
        plan_summary=plan_summary,
    )
    logger.info(
        "Scenario %s complete: cost=%.2f BDT, grid=%.2f kWh",
        request.scenario_id, total_cost_bdt, total_grid_kwh,
    )
    return response
