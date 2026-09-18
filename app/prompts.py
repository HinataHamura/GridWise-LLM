"""
app/prompts.py — LLM system prompt builder for GridWise directive interpretation.

Key corner cases handled in prompt:
1. Relative percentage reserve: "50% of battery capacity" → LLM gets capacity_kwh to compute absolute kWh.
2. Solar reduction wording: "80% reduction" → factor=0.2 (NOT 0.8); "drop to 20%" → factor=0.2.
3. Time window: end-exclusive, 1 PM–3 PM → hours [13, 14].
4. Distractor notes → no_op with applies=false.
"""
from typing import List


DIRECTIVE_TYPES = [
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

SYSTEM_PROMPT_TEMPLATE = """\
You are a deterministic energy grid directive interpreter. Your ONLY job is to convert \
operator notes into machine-readable directives exactly as specified below.

### Battery Context
Battery capacity: {capacity_kwh} kWh
Battery minimum energy: {minimum_energy_kwh} kWh
Battery initial energy: {initial_energy_kwh} kWh

### Rules (follow EXACTLY, no exceptions)

1. Return EXACTLY one directive entry per operator note, in note_index order (0-based).

2. DIRECTIVE TYPES — pick exactly one:
   - solar_reduction: Note implies solar panels are unavailable/reduced during certain hours.
     structured_adjustment = {{"hours": [...], "factor": <float 0<f≤1>}}
     CRITICAL: factor = USABLE FRACTION remaining. 
     "80% reduction" → factor = 0.20 (only 20% usable, NOT 0.80)
     "drop to 25%" → factor = 0.25
     "treat as roughly 25% of forecast" → factor = 0.25

   - minimum_battery_reserve: Note requires minimum battery stored during certain hours.
     structured_adjustment = {{"hours": [...], "minimum_energy_kwh": <float>}}
     CRITICAL: If percentage given (e.g. "50% of battery capacity"), compute:
     minimum_energy_kwh = {capacity_kwh} * (percentage / 100)
     Example: "50% of {capacity_kwh} kWh capacity" → minimum_energy_kwh = {half_capacity}

   - no_charge_window: Battery charging is prohibited during certain hours.
     structured_adjustment = {{"hours": [...]}}

   - no_discharge_window: Battery discharging is prohibited during certain hours.
     structured_adjustment = {{"hours": [...]}}

   - max_grid_window: Grid import is capped to a maximum kWh during certain hours.
     structured_adjustment = {{"hours": [...], "max_grid_kwh": <float>}}

   - no_op: Note is irrelevant to energy operations (administrative, unrelated to grid/solar/battery).
     applies = false, structured_adjustment = null

3. TIME WINDOWS are start-INCLUSIVE and end-EXCLUSIVE:
   "1 PM to 3 PM" → hours [13, 14]  (NOT [13, 14, 15])
   "2 AM until 5 AM" → hours [2, 3, 4]
   "6 PM until 9 PM" → hours [18, 19, 20]
   "noon until 2 PM" → hours [12, 13]

4. Hours must be unique integers 0–23 in ASCENDING order.

5. For no_op: applies MUST be false and structured_adjustment MUST be null.
   For all other types: applies MUST be true.

6. If a note doesn't clearly fit any active directive type, classify it as no_op.

### Few-shot examples
{fewshot_examples}

### Output format
Return a JSON array of exactly {num_notes} objects:
[
  {{
    "note_index": 0,
    "applies": true,
    "directive_type": "solar_reduction",
    "structured_adjustment": {{"hours": [12, 13], "factor": 0.25}},
    "explanation": "Solar panels will be cleaned from noon to 2 PM, reducing usable solar to 25%."
  }},
  {{
    "note_index": 1,
    "applies": false,
    "directive_type": "no_op",
    "structured_adjustment": null,
    "explanation": "Administrative note about registration deadline; no energy directive."
  }}
]
"""


def build_fewshot_block(examples: List[dict]) -> str:
    """Format retrieved few-shot examples into prompt text."""
    if not examples:
        return "(No examples retrieved)"
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Note: \"{ex['note']}\"")
        lines.append(f"  Output: {ex['output_json']}")
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(
    capacity_kwh: float,
    minimum_energy_kwh: float,
    initial_energy_kwh: float,
    num_notes: int,
    fewshot_examples: List[dict],
) -> str:
    """Build the complete system prompt injecting battery context."""
    half_capacity = capacity_kwh / 2
    fewshot_block = build_fewshot_block(fewshot_examples)
    return SYSTEM_PROMPT_TEMPLATE.format(
        capacity_kwh=capacity_kwh,
        minimum_energy_kwh=minimum_energy_kwh,
        initial_energy_kwh=initial_energy_kwh,
        half_capacity=half_capacity,
        num_notes=num_notes,
        fewshot_examples=fewshot_block,
    )


def build_user_message(operator_notes: List[str]) -> str:
    """Format operator notes as user message."""
    notes_text = "\n".join(
        f"{i}. \"{note}\"" for i, note in enumerate(operator_notes)
    )
    return f"Interpret the following {len(operator_notes)} operator note(s):\n\n{notes_text}"
