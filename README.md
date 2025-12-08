# Bike Sharing Analysis API

A FastAPI backend for analyzing bike sharing data with ML predictions.

## Quick Start with Docker Compose

```powershell

# OPTIONAL: Copy env file and adjust values as needed 
cp .env.example .env

# Start services
docker-compose up -d

# Train the models
docker-compose exec api python scripts/train_model.py
docker-compose exec api python scripts/train_model_full.py

# To stop: 
docker-compose down

```

## Example usage

###

#### Load dataset

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/load-dataset" -Method POST
```

#### Get statistics

```powershell
# Average rentals per hour
Invoke-RestMethod -Uri "http://localhost:8000/statistics?group_by=hour&metric=avg" | ConvertTo-Json -Depth 5

# Average rentals per weekday
Invoke-RestMethod -Uri "http://localhost:8000/statistics?group_by=weekday&metric=avg" | ConvertTo-Json -Depth 5

# Total rentals per season
Invoke-RestMethod -Uri "http://localhost:8000/statistics?group_by=season&metric=sum" | ConvertTo-Json -Depth 5

# With date filter
Invoke-RestMethod -Uri "http://localhost:8000/statistics?group_by=weekday&metric=avg&start_date=2011-01-01&end_date=2011-06-30" | ConvertTo-Json -Depth 5
```

#### Export statistics

Open in browser to download CSV:

<http://localhost:8000/export?group_by=hour&metric=avg>

#### Predictions

```powershell
# Simple prediction
Invoke-RestMethod -Uri "http://localhost:8000/predict?season=2&hr=17&weekday=3&workingday=1&weathersit=1" -Method POST | ConvertTo-Json -Depth 5

# Complex prediction
Invoke-RestMethod -Uri "http://localhost:8000/predict_complex?season=2&yr=1&mnth=6&hr=17&holiday=0&weekday=3&workingday=1&weathersit=1&temp=0.5&atemp=0.5&hum=0.5&windspeed=0.2" -Method POST | ConvertTo-Json -Depth 5
```

## API Endpoints

Documentation: <http://localhost:8000/docs>

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check (database + models) |
| POST | `/load-dataset` | Load CSV data into database |
| GET | `/statistics` | Aggregated statistics (group by hour/day/season/etc.) |
| POST | `/predict` | Predict rentals (5 features) |
| POST | `/predict_complex` | Predict rentals (12 features) |
| GET | `/export` | Download statistics as CSV |

## Dataset

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) - 17,379 hourly records from Washington D.C. (2011-2012).

## ML Models

Two *Random Forest Regressor* models are available:

| Endpoint | Features |
|----------|----------|
| `/predict` | season, hr, weekday, workingday, weathersit |
| `/predict_complex` | All above + yr, mnth, holiday, temp, atemp, hum, windspeed |

## Configuration

Environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_DB` | Database name |
| `API_PORT` | API port (default: 8000) |
