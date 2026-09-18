"""
app/graph.py — LangGraph state machine for LLM interpretation pipeline.

Flow:
  retrieve_fewshot
  → interpret_notes (LLM call)
  → guardrail_validate
      valid → finalize
      invalid (retries < MAX_RETRIES) → self_correct → interpret_notes
      exhausted → safe_fallback (all no_op)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.fewshot import retrieve_top_k
from app.llm import call_llm_chain
from app.prompts import build_system_prompt, build_user_message
from app.validators import deterministic_guardrail_validate, make_safe_fallback

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ─── Graph state ──────────────────────────────────────────────────────────────

class InterpretState(TypedDict):
    # Inputs
    operator_notes: List[str]
    capacity_kwh: float
    minimum_energy_kwh: float
    initial_energy_kwh: float

    # Working state
    fewshot_examples: List[dict]
    system_prompt: str
    user_message: str
    raw_directives: Optional[List[dict]]
    validation_errors: List[str]
    retry_count: int
    correction_hint: str

    # Output
    directive_interpretation: Optional[List[dict]]
    finalized: bool


# ─── Node functions ───────────────────────────────────────────────────────────

def node_retrieve_fewshot(state: InterpretState) -> Dict[str, Any]:
    """Retrieve similar few-shot examples for each note."""
    notes = state["operator_notes"]
    examples: List[dict] = []
    for note in notes:
        top = retrieve_top_k(note, k=2)
        # Deduplicate
        for ex in top:
            if ex not in examples:
                examples.append(ex)
    examples = examples[:5]  # cap total

    system_prompt = build_system_prompt(
        capacity_kwh=state["capacity_kwh"],
        minimum_energy_kwh=state["minimum_energy_kwh"],
        initial_energy_kwh=state["initial_energy_kwh"],
        num_notes=len(notes),
        fewshot_examples=examples,
    )
    user_message = build_user_message(notes)

    return {
        "fewshot_examples": examples,
        "system_prompt": system_prompt,
        "user_message": user_message,
        "retry_count": 0,
        "validation_errors": [],
        "correction_hint": "",
        "raw_directives": None,
        "directive_interpretation": None,
        "finalized": False,
    }


def node_interpret_notes(state: InterpretState) -> Dict[str, Any]:
    """Call LLM chain to get directive interpretation."""
    notes = state["operator_notes"]
    system_prompt = state["system_prompt"]
    user_message = state["user_message"]

    # On retries, append correction hint to user message
    hint = state.get("correction_hint", "")
    if hint:
        user_message = f"{user_message}\n\n[Correction required]: {hint}"

    try:
        raw = call_llm_chain(system_prompt, user_message, len(notes))
    except Exception as e:
        logger.error("LLM chain completely failed: %s", e)
        raw = None

    return {"raw_directives": raw}


def node_guardrail_validate(state: InterpretState) -> Dict[str, Any]:
    """Deterministically validate raw LLM output."""
    raw = state.get("raw_directives")
    num_notes = len(state["operator_notes"])

    if raw is None:
        return {
            "validation_errors": ["LLM returned no output"],
            "retry_count": state["retry_count"] + 1,
        }

    is_valid, errors = deterministic_guardrail_validate(raw, num_notes)
    if is_valid:
        return {"validation_errors": [], "raw_directives": raw}
    else:
        return {
            "validation_errors": errors,
            "retry_count": state["retry_count"] + 1,
            "correction_hint": f"Previous output had errors: {'; '.join(errors)}. Fix them and return a valid JSON array.",
        }


def node_finalize(state: InterpretState) -> Dict[str, Any]:
    """Accept validated directives."""
    return {
        "directive_interpretation": state["raw_directives"],
        "finalized": True,
    }


def node_self_correct(state: InterpretState) -> Dict[str, Any]:
    """Prepare for retry (state already updated with correction_hint)."""
    logger.info(
        "Self-correcting (retry %d/%d): %s",
        state["retry_count"],
        MAX_RETRIES,
        state["validation_errors"],
    )
    return {}


def node_safe_fallback(state: InterpretState) -> Dict[str, Any]:
    """Return all-no_op when retries exhausted."""
    logger.warning("Exhausted retries — using safe_fallback (all no_op)")
    return {
        "directive_interpretation": make_safe_fallback(len(state["operator_notes"])),
        "finalized": True,
    }


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_validate(state: InterpretState) -> str:
    if not state["validation_errors"]:
        return "finalize"
    if state["retry_count"] >= MAX_RETRIES:
        return "safe_fallback"
    return "self_correct"


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_interpretation_graph():
    graph = StateGraph(InterpretState)

    graph.add_node("retrieve_fewshot", node_retrieve_fewshot)
    graph.add_node("interpret_notes", node_interpret_notes)
    graph.add_node("guardrail_validate", node_guardrail_validate)
    graph.add_node("finalize", node_finalize)
    graph.add_node("self_correct", node_self_correct)
    graph.add_node("safe_fallback", node_safe_fallback)

    graph.set_entry_point("retrieve_fewshot")
    graph.add_edge("retrieve_fewshot", "interpret_notes")
    graph.add_edge("interpret_notes", "guardrail_validate")
    graph.add_conditional_edges(
        "guardrail_validate",
        route_after_validate,
        {
            "finalize": "finalize",
            "self_correct": "self_correct",
            "safe_fallback": "safe_fallback",
        },
    )
    graph.add_edge("self_correct", "interpret_notes")
    graph.add_edge("finalize", END)
    graph.add_edge("safe_fallback", END)

    return graph.compile()


# Compile once at module load
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_interpretation_graph()
    return _graph


def run_interpretation(
    operator_notes: List[str],
    capacity_kwh: float,
    minimum_energy_kwh: float,
    initial_energy_kwh: float,
) -> List[dict]:
    """
    Run the full LangGraph interpretation pipeline.
    Returns validated directive_interpretation list.
    """
    graph = get_graph()
    initial_state: InterpretState = {
        "operator_notes": operator_notes,
        "capacity_kwh": capacity_kwh,
        "minimum_energy_kwh": minimum_energy_kwh,
        "initial_energy_kwh": initial_energy_kwh,
        "fewshot_examples": [],
        "system_prompt": "",
        "user_message": "",
        "raw_directives": None,
        "validation_errors": [],
        "retry_count": 0,
        "correction_hint": "",
        "directive_interpretation": None,
        "finalized": False,
    }
    final_state = graph.invoke(initial_state)
    return final_state["directive_interpretation"]
