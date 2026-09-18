# GridWise - BUP CSE Fest 2026 Preliminary

LLM-assisted 24-hour smart-campus energy scheduling API for the BUP CSE Fest 2026 Hackathon preliminary round.

## Live Deployment & Docker Registry

- 🚀 **Live Production API:** [https://gridwise-api-live-production.up.railway.app](https://gridwise-api-live-production.up.railway.app)
  - `GET /health`: [https://gridwise-api-live-production.up.railway.app/health](https://gridwise-api-live-production.up.railway.app/health)
  - `POST /optimize-energy`: `https://gridwise-api-live-production.up.railway.app/optimize-energy`
- 🐳 **Docker Hub Repository:** [https://hub.docker.com/r/azthebest/gridwise-bup-2026](https://hub.docker.com/r/azthebest/gridwise-bup-2026)
- 📦 **GitHub Container Registry:** `ghcr.io/mainulasif/gridwise-bup-2026:latest`

## Verification & Submission Checklist

- [x] **Live API Health Check:** `GET https://gridwise-api-live-production.up.railway.app/health` returns `{"status":"ok"}`.
- [x] **Live Optimization API:** `POST https://gridwise-api-live-production.up.railway.app/optimize-energy` verified & passes public sample (matches 1262.0 BDT optimal cost).
- [x] **Docker Image:** GitHub Actions workflow builds & pushes fallback image to Docker Hub (`azthebest/gridwise-bup-2026:latest`) and GHCR.
- [x] **Security & Secret Safety:** No API keys, `.env` files, or secrets committed.
- [ ] **3-Minute Video:** Recorded & uploaded (script available at [VIDEO_SCRIPT.md](file:///d:/Download/GridWise_Submission/GridWise_Submission/VIDEO_SCRIPT.md)).
- [ ] **Repository Visibility:** Make repository **Public** immediately after the 11:00 PM submission deadline.

This repository implements the required pipeline:

**operator notes -> local generative language model -> deterministic guardrails/normalization -> linear optimizer -> independent replay validation -> exact JSON response**

The service exposes exactly:

- `GET /health`
- `POST /optimize-energy`

## Architecture

```text
POST /optimize-energy
        |
        v
Pydantic request validation
        |
        v
Local LLM: google/flan-t5-small
(language understanding + structured directive proposal)
        |
        v
Deterministic guardrails
- allowed directive type
- one note -> one interpretation
- whole-hour normalization
- applies/no_op semantics
- solar factor range
- reserve/grid-cap numeric validation
        |
        v
Directive application
        |
        v
SciPy HiGHS linear program
- grid, solar, signed battery flow, state-of-charge
- no simultaneous charge/discharge
- battery/rate/directive constraints
- end-of-day neutrality
- minimum grid-energy cost
        |
        v
Independent hour-by-hour replay validation
        |
        v
Exact response schema + recalculated totals
```

### Why the battery optimizer uses signed flow

A single battery-flow variable is used per hour: positive means discharge and negative means charge. This prevents simultaneous charge and discharge without introducing binary variables, keeping the problem a fast linear program suitable for repeated judge requests.

## LLM / model disclosure

- **Provider:** local model, no hosted API dependency
- **Model:** `google/flan-t5-small`
- **Role:** reads every `operator_notes` string and produces the semantic structured directive proposal used in the optimization path
- **Decoding:** deterministic (`do_sample=false`, one beam)
- **Runtime:** CPU
- **Secrets:** none required for the default deployment

The Docker image downloads the public model during image build and stores it inside the image. At runtime `TRANSFORMERS_OFFLINE=1` is enabled, so judge execution does not depend on an external LLM API, API key, quota, or model-host availability.

Deterministic code is used only after the language model to validate and normalize untrusted model output, as required by the challenge specification. It validates allowed directive types, time windows, percentages/factors, battery reserve values, grid caps, and `no_op` semantics before any directive reaches the optimizer.

## Supported directives

The implementation supports the complete official directive set:

- `solar_reduction`
- `minimum_battery_reserve`
- `no_charge_window`
- `no_discharge_window`
- `max_grid_window`
- `no_op`

Time windows use start-inclusive/end-exclusive whole-hour semantics. For example, `1 PM to 3 PM` maps to `[13, 14]`. For `solar_reduction`, the stored `factor` is the usable fraction remaining; an 80% reduction becomes `0.2`.

## Optimizer

The optimizer uses `scipy.optimize.linprog(method="highs")` and minimizes:

```text
sum(grid_kwh[h] * tariff_bdt_per_kwh[h]) for h = 0..23
```

while enforcing:

- hourly demand balance
- effective-solar availability
- non-negative grid import
- battery capacity and minimum reserve
- hourly charge/discharge rate limits
- `no_charge_window`
- `no_discharge_window`
- `minimum_battery_reserve`
- `max_grid_window`
- end-of-day battery neutrality

After solving, the returned schedule is replayed independently hour by hour. A plan is not returned if replay validation fails.

## Local quickstart - Docker (recommended)

Requirements: Docker with internet access during the first image build.

```bash
git clone https://github.com/mainulasif/gridwise-bup-2026.git
cd gridwise-bup-2026
docker build -t gridwise-bup-2026:local .
docker run --rm -p 8000:8000 gridwise-bup-2026:local
```

The first build downloads the local language model into the image. Runtime requires no API key.

Check readiness:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Local quickstart - Python

Python 3.11 is recommended. CPU PyTorch must be installed before the other dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
pip install -r requirements.txt

# For direct Python execution, use a Hugging Face model identifier or local model path.
export LOCAL_LLM_MODEL=google/flan-t5-small
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On Windows Command Prompt, set the model with:

```bat
set LOCAL_LLM_MODEL=google/flan-t5-small
```

## Environment variables

No secret environment variables are required by the default solution.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | HTTP listening port. Railway supplies this automatically. |
| `LOCAL_LLM_MODEL` | `google/flan-t5-small` in Python; `/opt/models/flan-t5-small` in Docker | Hugging Face model ID or local model directory. |
| `LLM_MAX_NEW_TOKENS` | `128` | Maximum generated tokens for one directive interpretation. |
| `LOG_LEVEL` | `INFO` | Application log level. |

Do not commit `.env` files, API keys, tokens, passwords, or other secrets.

## Example request

Use any official scenario request object. A representative official public sample fixture is included at `tests/data/sample01.json`; the same endpoint can be exercised with the full organizer public pack.

For example, save a case input to `request.json`, then run:

```bash
curl -sS -X POST http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  --data @request.json
```

Successful responses contain exactly the required top-level fields:

```text
scenario_id
directive_interpretation
hourly_plan
total_grid_kwh
total_cost_bdt
peak_grid_kwh
plan_summary
```

## Public sample validation

A representative official public sample fixture is stored at:

```text
tests/data/sample01.json
```

With the service running:

```bash
python scripts/run_public_sample.py --base-url http://127.0.0.1:8000
```

The script verifies machine-checkable directive semantics and the official optimal cost for SAMPLE-01. Equivalent optimal hourly schedules are allowed by the official specification, so it does not require byte-for-byte schedule equality.

The deterministic optimizer and normalization layers also have unit tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

Before push, the optimizer was also checked locally against all 10 official public scenarios and matched the organizer optimal cost in every case. The committed unit suite covers the representative fixture plus request-validation behavior.

## API behavior

### `GET /health`

Returns HTTP 200 after the local language model has loaded:

```json
{"status":"ok"}
```

### `POST /optimize-energy`

- Accepts the exact official scenario schema.
- Requires exactly 24 unique hourly records, hours 0 through 23.
- Requires 1-3 non-empty operator notes.
- Returns one `directive_interpretation` entry per note in `note_index` order.
- Returns exactly 24 `hourly_plan` entries.
- Recalculates all reported totals from the returned plan.

Controlled error handling:

- malformed JSON -> HTTP 400
- structurally/semantically invalid request -> HTTP 422
- malformed/unsupported local-model interpretation -> controlled HTTP 500 with no stack trace or secret data in the response
- infeasible/invalid schedule -> controlled HTTP 422

## Guardrails

Language-model output is never applied directly as mathematical constraints. Before optimization, the service verifies:

- directive type is one of the six official values
- every non-`no_op` directive has `applies=true`
- `no_op` has `applies=false` and `structured_adjustment=null`
- hours are unique integers in `0..23`, sorted ascending
- solar factor is finite and in `[0, 1]`
- reserve is finite, non-negative, and no greater than battery capacity
- grid cap is finite and non-negative
- no base demand, tariff, or battery parameter is invented or modified

When compatible directives overlap, the optimizer uses the most restrictive combination (minimum solar availability, highest reserve, lowest grid cap, and union of charge/discharge prohibition windows), ensuring all returned hard directives are satisfied.

## Reliability / performance choices

- Local model removes hosted-provider rate limits and credential failure modes.
- Model is loaded and warmed before `/health` becomes ready.
- Inference uses deterministic decoding.
- The optimizer is a small 24-hour LP solved with HiGHS.
- A final replay validator independently checks the schedule before response serialization.
- No raw exception stack traces are sent to clients.

## Docker fallback images

A GitHub Actions workflow at `.github/workflows/docker-ghcr.yml` builds and pushes the Docker image to **both** GitHub Container Registry and Docker Hub on every push to `main`:

```text
ghcr.io/mainulasif/gridwise-bup-2026:latest
docker.io/azthebest/gridwise-bup-2026:latest
```

- **Docker Hub URL:** [https://hub.docker.com/r/azthebest/gridwise-bup-2026](https://hub.docker.com/r/azthebest/gridwise-bup-2026)

After the official submission deadline, make the repository public as required by the event rules and ensure the packages are publicly pullable for judge fallback access.

Example fallback commands:

**From GitHub Container Registry:**
```bash
docker pull ghcr.io/mainulasif/gridwise-bup-2026:latest
docker run --rm -p 8000:8000 ghcr.io/mainulasif/gridwise-bup-2026:latest
curl http://127.0.0.1:8000/health
```

**From Docker Hub:**
```bash
docker pull azthebest/gridwise-bup-2026:latest
docker run --rm -p 8000:8000 azthebest/gridwise-bup-2026:latest
curl http://127.0.0.1:8000/health
```

## Cloud Deployment (Free & Paid Options)

The application is stateless and containerized, making it easy to deploy to any Docker-compatible cloud host.

### 1. Hugging Face Spaces (Highly Recommended / Best Free Option)
Hugging Face Spaces provides a generous free tier (16 GB RAM) perfect for running the local PyTorch model without Out-Of-Memory (OOM) crashes.
- Create a New Space and select **Docker** as the SDK.
- Connect your GitHub repository to auto-deploy.
- **Important Tweak:** Hugging Face exposes port `7860`. Go to your Space **Settings -> Variables and secrets**, and add a New Variable: `PORT = 7860`.
- Your app will be live at a public `.hf.space` URL.

### 2. Render.com (Free Tier)
You can deploy for free as a Docker Web Service on Render. 
- Connect your GitHub repository and select the **Free** tier. 
- *Caveat:* The free tier is limited to 512 MB RAM. Since the app runs a language model, it might experience OOM (Out Of Memory) errors during deployment. Also, the instance sleeps after 15 minutes of inactivity, causing ~1-minute cold starts.

### 3. Railway (If you have active credits)
`railway.toml` configures Dockerfile builds, `/health` readiness checking, and restart behavior. Railway supplies `PORT` automatically.
The repository contains no deployment-time secrets. Deploy the repository as one service, then generate one public Railway domain and submit that base URL.

## Repository structure

```text
app/
  main.py          FastAPI endpoints and controlled error handling
  models.py        exact request/response validation models
  interpreter.py   local LLM + deterministic interpretation guardrails
  optimizer.py     directive application, LP solver, replay validator
scripts/
  run_public_sample.py
tests/
  data/sample01.json
  test_public_samples.py
  test_validation.py
.github/workflows/
  docker-ghcr.yml
Dockerfile
railway.toml
requirements.txt
requirements-dev.txt
pyproject.toml
VIDEO_SCRIPT.md
SUBMISSION_CHECKLIST.md
```

## Known limitations

- The local language model is intentionally small to keep deployment fast and self-contained. Deterministic normalization improves consistency for whole-hour expressions and numeric values, but semantic interpretation still depends on the local generative model and can be challenged by unusually indirect language.
- The challenge specification does not define special composition semantics for overlapping directives of the same type. This implementation applies the most restrictive interpretation, prioritizing hard-constraint validity.
- The first Docker build is larger/slower because it installs CPU PyTorch and embeds the language model. Runtime startup is substantially faster and does not require model download.

## External tools / dependencies

Core implementation logic in this repository is original for this challenge. External libraries/models used include FastAPI, Pydantic, SciPy/HiGHS, NumPy, Hugging Face Transformers, PyTorch CPU, SentencePiece, Uvicorn, and `google/flan-t5-small`.

No challenge inputs, expected outputs, or reference schedules are hard-coded into the runtime service.
