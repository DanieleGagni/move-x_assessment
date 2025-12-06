# Bike Sharing Analysis API

A FastAPI backend for analyzing bike sharing data with ML predictions.

## Quick Start with Docker Compose

```bash
# Start services
docker-compose up -d

# Train the models
docker-compose exec api python scripts/train_model.py
docker-compose exec api python scripts/train_model_full.py

# Load dataset
curl -X POST http://localhost:8000/load-dataset

# Access API docs
http://localhost:8000/docs
```

To stop: `docker-compose down`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check (database + models) |
| POST | `/load-dataset` | Load CSV data into database |
| GET | `/statistics` | Aggregated statistics (group by hour/day/season/etc.) |
| POST | `/predict` | Predict rentals (5 features) |
| POST | `/predict_complex` | Predict rentals (12 features) |
| GET | `/export` | Download statistics as CSV |

## ML Models

Both models use Random Forest Regressor.

| Endpoint | Features |
|----------|----------|
| `/predict` | season, hr, weekday, workingday, weathersit |
| `/predict_complex` | All above + yr, mnth, holiday, temp, atemp, hum, windspeed |

Training:
```bash
python scripts/train_model.py        # Simple model
python scripts/train_model_full.py   # Full model
```

## Project Structure

```
app/
  main.py           # FastAPI endpoints
  database.py       # Database connection
  models.py         # SQLAlchemy ORM model
  model_inference.py # ML prediction logic
scripts/
  train_model.py      # Train simple model
  train_model_full.py # Train full model
models/
  bike_rental_model.joblib       # Simple model
  bike_rental_model_full.joblib  # Full model
data/
  hour.csv          # UCI Bike Sharing Dataset
```

## Dataset

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) - 17,379 hourly records from Washington D.C. (2011-2012).

## Configuration

Environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_DB` | Database name |
| `API_PORT` | API port (default: 8000) |
