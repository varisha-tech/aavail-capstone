# AAVAIL Capstone — AI in Production
**IBM AI Workflow Capstone Project**

A production-ready ML API that predicts AAVAIL revenue by country using time-series features and multiple compared models.

---

## Project Structure

```
aavail-capstone/
├── api/
│   └── app.py              # Flask REST API
├── model/
│   ├── model.py            # Training, loading, prediction
│   └── monitor.py          # Performance monitoring
├── tests/
│   ├── test_api.py         # Unit tests — API
│   ├── test_model.py       # Unit tests — Model
│   └── test_logging.py     # Unit tests — Logging (isolated)
├── notebooks/
│   └── EDA.ipynb           # EDA with visualizations + model comparison
├── scripts/
│   └── ingest.py           # Data ingestion script (automation-ready)
├── logs/                   # API and monitor logs
├── data/                   # Place aavail-target.csv here
├── run_tests.py            # Run ALL tests with single script
├── Dockerfile              # Docker containerization
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
python model/model.py
```

### 3. Run all unit tests (single script)
```bash
python run_tests.py
```

### 4. Start the API
```bash
python api/app.py
```

### 5. Test the API
```bash
# Health check
curl http://localhost:8080/ping

# Predict for a specific country
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "united_states", "year": 2018, "month": 6}'

# Predict for all countries combined
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "all", "year": 2018, "month": 6}'

# View metrics
curl http://localhost:8080/metrics

# View logs
curl http://localhost:8080/logs
```

---

## Docker

```bash
# Build
docker build -t aavail-capstone .

# Run
docker run -p 8080:8080 aavail-capstone

# Test inside container
docker run aavail-capstone python run_tests.py
```

---

## API Endpoints

| Method | Endpoint   | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /ping     | Health check                         |
| POST   | /predict  | Predict revenue for country + month  |
| POST   | /train    | Retrain all models                   |
| GET    | /metrics  | Model RMSE metrics (all countries)   |
| GET    | /logs     | Recent API log entries               |

### /predict payload
```json
{
  "country": "united_states",
  "year": 2018,
  "month": 6
}
```
Valid countries: `all`, `united_states`, `united_kingdom`, `france`, `germany`, `japan`

---

## Peer Review Checklist

| Criterion | Status |
|-----------|--------|
| Unit tests for the API | ✅ `tests/test_api.py` |
| Unit tests for the model | ✅ `tests/test_model.py` |
| Unit tests for logging | ✅ `tests/test_logging.py` |
| All tests run with single script | ✅ `python run_tests.py` |
| Performance monitoring mechanism | ✅ `model/monitor.py` + `/metrics` endpoint |
| Read/write unit tests isolated from production | ✅ `test_logging.py` uses `tempfile` |
| API works for specific country + all combined | ✅ `/predict` supports all countries |
| Data ingestion as function/script | ✅ `model/model.py::ingest_data()` + `scripts/ingest.py` |
| Multiple models compared | ✅ GradientBoosting, RandomForest, Ridge |
| EDA with visualizations | ✅ `notebooks/EDA.ipynb` |
| Containerized in Docker | ✅ `Dockerfile` |
| Visualization comparing model vs baseline | ✅ `EDA.ipynb` — final cell |
