"""
app/fewshot.py — Few-shot retrieval for directive interpretation.

Strategy:
1. Try HuggingFace Inference API (hosted all-MiniLM-L6-v2) for semantic embeddings.
2. Fall back to Jaccard keyword overlap if HF fails/rate-limits.

All 10 public sample cases are hardcoded as a retrieval corpus.
Embeddings are cached in-memory on first call.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ─── Few-shot corpus (from 10 public sample cases) ───────────────────────────

FEWSHOT_CORPUS = [
    {
        "note": "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "solar_reduction", "structured_adjustment": {"hours": [12, 13], "factor": 0.25}, "explanation": "Solar panels being cleaned noon to 2 PM; only 25% usable."}',
    },
    {
        "note": "The sports office moved next month's registration deadline.",
        "output_json": '{"note_index": 0, "applies": false, "directive_type": "no_op", "structured_adjustment": null, "explanation": "Administrative note unrelated to energy operations."}',
    },
    {
        "note": "The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "no_charge_window", "structured_adjustment": {"hours": [2, 3, 4]}, "explanation": "Charger isolated from 2 AM to 5 AM; no charging during hours 2, 3, 4."}',
    },
    {
        "note": "Keep at least 50% of the battery capacity stored in the battery from 6 PM until 9 PM for emergency operations.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "minimum_battery_reserve", "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0}, "explanation": "50% of 200 kWh capacity = 100 kWh minimum reserve from 6 PM to 9 PM."}',
    },
    {
        "note": "Do not use the battery to supply loads from 6 PM to 8 PM; the discharge circuit will be under inspection.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "no_discharge_window", "structured_adjustment": {"hours": [18, 19]}, "explanation": "Discharge circuit under inspection from 6 PM to 8 PM; no discharge hours 18-19."}',
    },
    {
        "note": "A temporary feeder fault will limit grid import to at most 155 kWh per hour from 6 PM until 9 PM.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "max_grid_window", "structured_adjustment": {"hours": [18, 19, 20], "max_grid_kwh": 155.0}, "explanation": "Grid import capped at 155 kWh/h from 6 PM to 9 PM due to feeder fault."}',
    },
    {
        "note": "Solar output will drop by 50% between 10 AM and 1 PM due to shading from nearby construction.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "solar_reduction", "structured_adjustment": {"hours": [10, 11, 12], "factor": 0.5}, "explanation": "50% reduction means factor=0.5 usable from 10 AM to 1 PM."}',
    },
    {
        "note": "Keep a 90 kWh reserve in the battery between 6 PM and 10 PM.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "minimum_battery_reserve", "structured_adjustment": {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 90.0}, "explanation": "90 kWh minimum reserve required from 6 PM to 10 PM."}',
    },
    {
        "note": "Grid import must not exceed 180 kWh per hour from 7 PM until 9 PM due to transformer stress.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "max_grid_window", "structured_adjustment": {"hours": [19, 20], "max_grid_kwh": 180.0}, "explanation": "Grid capped at 180 kWh/h from 7 PM to 9 PM."}',
    },
    {
        "note": "Solar panels will lose about 80% of their output from 11 AM to 2 PM due to heavy cloud cover.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "solar_reduction", "structured_adjustment": {"hours": [11, 12, 13], "factor": 0.2}, "explanation": "80% reduction means only 20% remains usable; factor=0.2 for hours 11-13."}',
    },
    {
        "note": "Battery charging is not permitted from 11 AM to 1 PM due to inverter servicing.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "no_charge_window", "structured_adjustment": {"hours": [11, 12]}, "explanation": "No charging from 11 AM to 1 PM during inverter servicing."}',
    },
    {
        "note": "Battery discharge is blocked from 5 PM to 7 PM for safety checks.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "no_discharge_window", "structured_adjustment": {"hours": [17, 18]}, "explanation": "No discharging from 5 PM to 7 PM for safety checks."}',
    },
    {
        "note": "The IT manager approved the new server rack installation.",
        "output_json": '{"note_index": 0, "applies": false, "directive_type": "no_op", "structured_adjustment": null, "explanation": "Administrative note about IT infrastructure; no energy directive."}',
    },
    {
        "note": "Maintain at least 80 kWh in the battery from 6 PM to 10 PM.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "minimum_battery_reserve", "structured_adjustment": {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 80.0}, "explanation": "80 kWh minimum reserve required from 6 PM to 10 PM."}',
    },
    {
        "note": "Grid supply will be restricted to no more than 190 kWh per hour from 7 PM to 9 PM.",
        "output_json": '{"note_index": 0, "applies": true, "directive_type": "max_grid_window", "structured_adjustment": {"hours": [19, 20], "max_grid_kwh": 190.0}, "explanation": "Grid capped at 190 kWh/h from 7 PM to 9 PM."}',
    },
]

# ─── Embedding cache ──────────────────────────────────────────────────────────

_corpus_embeddings: Optional[np.ndarray] = None
_hf_available: Optional[bool] = None

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"


def _get_hf_headers() -> dict:
    token = os.environ.get("HF_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _embed_hf(texts: List[str]) -> Optional[np.ndarray]:
    """Call HF Inference API for embeddings. Returns None on failure."""
    try:
        resp = requests.post(
            HF_API_URL,
            headers=_get_hf_headers(),
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("HF embedding API returned %s", resp.status_code)
            return None
        data = resp.json()
        # data is list of embeddings (list of list of float)
        return np.array(data, dtype=np.float32)
    except Exception as e:
        logger.warning("HF embedding call failed: %s", e)
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and matrix b (rows are vectors)."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def _jaccard_similarity(query: str, doc: str) -> float:
    """Keyword overlap (Jaccard) fallback similarity."""
    q_words = set(re.findall(r"\w+", query.lower()))
    d_words = set(re.findall(r"\w+", doc.lower()))
    if not q_words and not d_words:
        return 0.0
    intersection = q_words & d_words
    union = q_words | d_words
    return len(intersection) / len(union)


def _warm_up_embeddings() -> None:
    """Pre-compute corpus embeddings on first call."""
    global _corpus_embeddings, _hf_available
    if _corpus_embeddings is not None:
        return
    corpus_texts = [ex["note"] for ex in FEWSHOT_CORPUS]
    embs = _embed_hf(corpus_texts)
    if embs is not None and embs.shape[0] == len(corpus_texts):
        _corpus_embeddings = embs
        _hf_available = True
        logger.info("HF embeddings cached for %d corpus examples", len(corpus_texts))
    else:
        _hf_available = False
        logger.info("Using Jaccard fallback for few-shot retrieval")


def retrieve_top_k(note: str, k: int = 3) -> List[dict]:
    """Retrieve top-k most similar few-shot examples for a given operator note."""
    _warm_up_embeddings()

    if _hf_available and _corpus_embeddings is not None:
        q_emb = _embed_hf([note])
        if q_emb is not None and q_emb.shape[0] == 1:
            sims = _cosine_similarity(q_emb[0], _corpus_embeddings)
            top_indices = np.argsort(sims)[::-1][:k]
            return [FEWSHOT_CORPUS[i] for i in top_indices]

    # Jaccard fallback
    scores = [_jaccard_similarity(note, ex["note"]) for ex in FEWSHOT_CORPUS]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [FEWSHOT_CORPUS[i] for i in top_indices]
