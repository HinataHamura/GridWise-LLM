# GridWise LLM

GridWise LLM is an energy optimization API for a 24-hour facility schedule. It interprets operator notes, applies operational constraints, and returns a validated battery, solar, and grid usage plan.

## Live Deployment

- API: https://doc2026-09-1821-35-33git.vercel.app
- Swagger UI: https://doc2026-09-1821-35-33git.vercel.app/docs
- Health check: https://doc2026-09-1821-35-33git.vercel.app/health
- Source code: https://github.com/HinataHamura/GridWise-LLM

## API Endpoints

### `GET /`
Returns service information and links to the API resources.

### `GET /health`
Returns the service health status.

### `POST /optimize-energy`
Interprets 1-3 operator notes and optimizes a complete 24-hour energy schedule.

The request must contain:

- `scenario_id`: unique scenario name
- `operator_notes`: 1-3 non-empty operational notes
- `hours`: exactly one entry for every hour from 0 to 23
- `battery`: battery capacity, reserve, and charge/discharge limits

Example request:

```json
{
  "scenario_id": "DEMO-001",
  "operator_notes": [
    "Solar output will drop by 50% between 10 AM and 1 PM due to shading."
  ],
  "hours": [
    {
      "hour": 0,
      "demand_kwh": 90,
      "solar_kwh": 0,
      "tariff_bdt_per_kwh": 6
    }
  ],
  "battery": {
    "capacity_kwh": 200,
    "initial_energy_kwh": 100,
    "minimum_energy_kwh": 20,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 50
  }
}
```

The `hours` array must include all 24 hourly objects. The response includes the interpreted directives, a 24-hour `hourly_plan`, total grid usage, total cost, peak grid usage, and a plan summary.

Example request with cURL:

```bash
curl -X POST https://doc2026-09-1821-35-33git.vercel.app/optimize-energy \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Supported Directives

The interpretation layer supports:

- Solar reduction windows
- Minimum battery reserve windows
- No-charge windows
- No-discharge windows
- Maximum grid import windows
- Irrelevant notes as `no_op`

All returned plans are checked against energy balance, battery limits, directive constraints, cost totals, and end-of-day battery neutrality.

## Run Locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn api.index:app --reload
```

Then open http://127.0.0.1:8000/docs.

For local LLM interpretation, configure provider keys in a `.env` file. Never commit `.env` or expose API keys publicly.

```env
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
HF_API_TOKEN=your_token
```

## Testing

```bash
pytest -q
```

## Deployment

The project is configured for Vercel's Python runtime. `api/index.py` exposes the FastAPI application and `vercel.json` preserves public API paths during rewrites.

## Project Structure

```text
api/index.py          Vercel ASGI entrypoint
app/main.py           FastAPI application and routes
app/graph.py          Interpretation workflow
app/llm.py            LLM provider chain
app/optimizer.py      Battery and grid optimization
app/validators.py     Deterministic directive validation
app/final_validator.py Plan replay validation
tests/                Unit and public-case tests
```
