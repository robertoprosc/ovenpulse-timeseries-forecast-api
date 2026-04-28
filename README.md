# OvenPulse Time-Series Forecasting API

Production-style academic project for bakery process analytics:
- anomaly detection over baking phase measurements
- statistical preprocessing and phase-impact analysis
- multi-model time-series forecasting (AR, ARIMA, SARIMA, Holt-Winters, Prophet)
- trend detection and automatic plot generation

This repository is portfolio-focused and intentionally excludes private/raw datasets from version control.

## Project structure

```text
ovenpulse-timeseries-forecast-api/
|-- app/
|   |-- models/
|   |-- services/
|   |-- utils/
|   `-- views/
|-- common/
|   `-- logger/
|-- docs/
|   `-- api_payload_examples.md
|-- resources/
|   |-- requests/              # Postman collection
|   `-- results/               # generated plots/statistics (ignored by git)
|-- tests/
|   `-- test_forecasting.py
|-- logs/
|   `-- .gitkeep
|-- application.py
|-- config.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Tech stack

- FastAPI + Uvicorn
- Pandas, NumPy, SciPy, scikit-learn
- Statsmodels + Prophet
- Matplotlib + Seaborn
- Pytest

## Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd /path/to/ovenpulse-timeseries-forecast-api
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run API

```bash
python application.py -host 127.0.0.1 -port 5003
```

Swagger docs:
- `http://127.0.0.1:5003/docs`

## Main endpoints

- `POST /stats/preprocess_and_calculate_stats`
- `POST /stats/affects_backing_time`
- `POST /anomaly/anomaly_detection_single`
- `POST /anomaly/anomaly_detection_massive`
- `POST /anomaly/anomaly_trend_detection`
- `POST /anomaly/forecasting_bakery_time`
- `POST /anomaly/forecasting_bakery_time_phases`

Postman collection:
- `resources/requests/timeseries-forecast-api.postman_collection.json`
- `docs/api_payload_examples.md` contains sample request bodies.

## Test

```bash
pytest -q
```

## Dataset policy

The folder `resources/dataset/` is ignored on purpose (see `.gitignore`).

To run the project locally:
1. Create `resources/dataset/`
2. Add your CSV/XLSX files there (expected by payloads and routes)
3. Run preprocessing endpoints first to generate fresh stats in `resources/results/`

## Suggested GitHub repo name

`ovenpulse-timeseries-forecast-api`
