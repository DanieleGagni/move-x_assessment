from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text, insert
from contextlib import asynccontextmanager
from datetime import date
import pandas as pd
import io
from typing import Optional
from pathlib import Path

from app.database import get_db, init_db
from app.models import BikeRental
from app.model_inference import predictor, predictor_complex


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load model on startup"""
    print("Starting up...")
    init_db()
    print("Application ready!")
    yield
    # Cleanup on shutdown (if needed)
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Bike Sharing Analysis API",
    description="API for analyzing bike sharing data with ML predictions",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== ROOT ENDPOINT ====================

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Bike Sharing Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "load_dataset": "POST /load-dataset",
            "statistics": "GET /statistics",
            "predict": "POST /predict (5 features)",
            "predict_complex": "POST /predict_complex (12 features)",
            "export": "GET /export",
            "health": "GET /health"
        }
    }

@app.get("/hello")
def hello_world():
    """Simple hello world endpoint for testing purposes"""
    return {"message": "Hello World!"}


# ==================== LOAD DATASET ENDPOINT ====================

@app.post("/load-dataset")
def load_dataset(
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Load bike sharing dataset into the database.
    If data is already present, it is reloaded.
    
    Options:
    1. Upload a CSV file
    2. Load from default location (data/hour.csv)
    
    Returns count of records loaded.
    """
    try:
        # Determine data source
        if file:
            # Option 1: Load from uploaded file
            df = pd.read_csv(file.file)
            source = f"uploaded file: {file.filename}"
        else:
            # Option 2: Load from default location
            csv_path = Path("data/hour.csv")
            if not csv_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Default dataset not found at {csv_path}. Please upload a file."
                )
            df = pd.read_csv(csv_path)
            source = str(csv_path)
        
        # Validate required columns
        required_columns = ['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 
                          'weekday', 'workingday', 'weathersit', 'temp', 
                          'atemp', 'hum', 'windspeed', 'cnt']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert date column to proper datetime format
        df['dteday'] = pd.to_datetime(df['dteday'])
        
        # Check if data already exists
        existing_count = db.query(BikeRental).count()
        
        if existing_count > 0:
            # Clear existing data
            db.query(BikeRental).delete()
            db.commit()
            print(f"Cleared {existing_count} existing records")
        
        # Convert DataFrame to list of dictionaries with string keys
        records: list[dict[str, any]] = df.to_dict('records')  # type: ignore
        
        # Bulk insert into database
        db.execute(insert(BikeRental), records)
        db.commit()
        
        record_count = len(records)
        
        print(f"Successfully loaded {record_count} records from {source}")
        
        return {
            "status": "success",
            "message": f"Successfully loaded dataset into database",
            "source": source,
            "records_loaded": record_count,
            "date_range": {
                "start": str(df['dteday'].min().date()),
                "end": str(df['dteday'].max().date())
            }
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error loading dataset: {str(e)}"
        )


# ==================== SHARED STATISTICS FUNCTION ====================

def calculate_statistics(
    db: Session,
    group_by: str,
    metric: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filter_season: Optional[int] = None,
    filter_weather: Optional[int] = None,
    filter_holiday: Optional[int] = None,
    filter_workingday: Optional[int] = None,
    filter_hour: Optional[int] = None
) -> dict:
    """
    Shared function to calculate aggregated statistics.
    Used by both /statistics and /export endpoints.
    
    Returns a dictionary with query info, summary, and results.
    """
    # Check if data exists
    total_records = db.query(BikeRental).count()
    if total_records == 0:
        raise HTTPException(
            status_code=404,
            detail="No data found. Please load dataset first using POST /load-dataset"
        )
    
    # Build base query with filters
    query = db.query(BikeRental)
    
    # Apply date filters
    if start_date:
        query = query.filter(BikeRental.dteday >= start_date)
    if end_date:
        query = query.filter(BikeRental.dteday <= end_date)
    
    # Apply categorical filters
    if filter_season is not None:
        query = query.filter(BikeRental.season == filter_season)
    if filter_weather is not None:
        query = query.filter(BikeRental.weathersit == filter_weather)
    if filter_holiday is not None:
        query = query.filter(BikeRental.holiday == filter_holiday)
    if filter_workingday is not None:
        query = query.filter(BikeRental.workingday == filter_workingday)
    if filter_hour is not None:
        query = query.filter(BikeRental.hr == filter_hour)
    
    # Check if any data matches filters
    filtered_count = query.count()
    if filtered_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No data found matching the specified filters"
        )
    
    # Determine the aggregation function
    metric_funcs = {
        "avg": func.avg(BikeRental.cnt),
        "sum": func.sum(BikeRental.cnt),
        "min": func.min(BikeRental.cnt),
        "max": func.max(BikeRental.cnt),
        "count": func.count(BikeRental.cnt)
    }
    agg_func = metric_funcs[metric]
    
    # Determine the grouping column and labels
    group_configs = {
        "hour": {
            "column": BikeRental.hr,
            "labels": {i: f"{i}:00" for i in range(24)},
            "label_key": "hour"
        },
        "weekday": {
            "column": BikeRental.weekday,
            "labels": {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 
                      4: "Thursday", 5: "Friday", 6: "Saturday"},
            "label_key": "weekday"
        },
        "month": {
            "column": BikeRental.mnth,
            "labels": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                      7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"},
            "label_key": "month"
        },
        "season": {
            "column": BikeRental.season,
            "labels": {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"},
            "label_key": "season"
        },
        "weather": {
            "column": BikeRental.weathersit,
            "labels": {1: "Clear/Partly Cloudy", 2: "Mist/Cloudy", 
                      3: "Light Snow/Rain", 4: "Heavy Rain/Snow"},
            "label_key": "weather"
        },
        "holiday": {
            "column": BikeRental.holiday,
            "labels": {0: "Non-Holiday", 1: "Holiday"},
            "label_key": "holiday"
        },
        "workingday": {
            "column": BikeRental.workingday,
            "labels": {0: "Non-Working Day", 1: "Working Day"},
            "label_key": "workingday"
        }
    }
    
    config = group_configs[group_by]
    group_column = config["column"]
    labels = config["labels"]
    label_key = config["label_key"]
    
    # Build and execute grouped query
    grouped = db.query(
        group_column,
        agg_func.label('value')
    ).filter(
        # Apply same filters to grouped query
        *([BikeRental.dteday >= start_date] if start_date else []),
        *([BikeRental.dteday <= end_date] if end_date else []),
        *([BikeRental.season == filter_season] if filter_season is not None else []),
        *([BikeRental.weathersit == filter_weather] if filter_weather is not None else []),
        *([BikeRental.holiday == filter_holiday] if filter_holiday is not None else []),
        *([BikeRental.workingday == filter_workingday] if filter_workingday is not None else []),
        *([BikeRental.hr == filter_hour] if filter_hour is not None else []),
    ).group_by(group_column).order_by(group_column).all()
    
    # Format results
    grouped_data = []
    for key, value in grouped:
        result = {
            label_key: labels.get(key, str(key)),
            f"{label_key}_code": key,
            f"{metric}_rentals": round(value, 2) if metric in ["avg"] else int(value)
        }
        grouped_data.append(result)
    
    # Calculate overall statistics for the filtered dataset
    overall_stats = {
        "total_rentals": query.with_entities(func.sum(BikeRental.cnt)).scalar() or 0,
        "avg_rentals": round(query.with_entities(func.avg(BikeRental.cnt)).scalar() or 0, 2),
        "max_rentals": query.with_entities(func.max(BikeRental.cnt)).scalar() or 0,
        "min_rentals": query.with_entities(func.min(BikeRental.cnt)).scalar() or 0,
        "record_count": filtered_count
    }
    
    # Build active filters summary
    active_filters = {
        "start_date": start_date,
        "end_date": end_date,
        "season": filter_season,
        "weather": filter_weather,
        "holiday": filter_holiday,
        "workingday": filter_workingday,
        "hour": filter_hour
    }
    # Remove None values
    active_filters = {k: v for k, v in active_filters.items() if v is not None}
    
    return {
        "query": {
            "group_by": group_by,
            "metric": metric,
            "filters": active_filters
        },
        "summary": overall_stats,
        "results": grouped_data
    }


# ==================== STATISTICS ENDPOINT ====================

@app.get("/statistics")
def get_statistics(
    group_by: str = Query("hour", enum=["hour", "weekday", "month", "season", "weather", "holiday", "workingday"]),
    metric: str = Query("avg", enum=["avg", "sum", "min", "max", "count"]),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    filter_season: Optional[int] = Query(None, ge=1, le=4, description="Filter by season (1:spring, 2:summer, 3:fall, 4:winter)"),
    filter_weather: Optional[int] = Query(None, ge=1, le=4, description="Filter by weather (1:clear, 2:mist, 3:light rain/snow, 4:heavy)"),
    filter_holiday: Optional[int] = Query(None, ge=0, le=1, description="Filter by holiday (0:no, 1:yes)"),
    filter_workingday: Optional[int] = Query(None, ge=0, le=1, description="Filter by working day (0:no, 1:yes)"),
    filter_hour: Optional[int] = Query(None, ge=0, le=23, description="Filter by hour (0-23)"),
    db: Session = Depends(get_db)
):
    """
    Get aggregated statistics about bike rentals with flexible grouping and filtering.
    
    Parameters:
    - group_by: Group results by hour/weekday/month/season/weather/holiday/workingday
    - metric: Aggregation function (avg/sum/min/max/count)
    - start_date: Filter from this date (optional)
    - end_date: Filter until this date (optional)
    - filter_season: Filter by season 1-4 (optional)
    - filter_weather: Filter by weather 1-4 (optional)
    - filter_holiday: Filter by holiday 0/1 (optional)
    - filter_workingday: Filter by working day 0/1 (optional)
    - filter_hour: Filter by hour 0-23 (optional)
    
    Returns aggregated statistics based on selected grouping and metric.
    """
    try:
        return calculate_statistics(
            db=db,
            group_by=group_by,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            filter_season=filter_season,
            filter_weather=filter_weather,
            filter_holiday=filter_holiday,
            filter_workingday=filter_workingday,
            filter_hour=filter_hour
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating statistics: {str(e)}"
        )


# ==================== EXPORT ENDPOINT ====================

@app.get("/export")
def export_statistics(
    group_by: str = Query("hour", enum=["hour", "weekday", "month", "season", "weather", "holiday", "workingday"]),
    metric: str = Query("avg", enum=["avg", "sum", "min", "max", "count"]),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    filter_season: Optional[int] = Query(None, ge=1, le=4, description="Filter by season (1:spring, 2:summer, 3:fall, 4:winter)"),
    filter_weather: Optional[int] = Query(None, ge=1, le=4, description="Filter by weather (1:clear, 2:mist, 3:light rain/snow, 4:heavy)"),
    filter_holiday: Optional[int] = Query(None, ge=0, le=1, description="Filter by holiday (0:no, 1:yes)"),
    filter_workingday: Optional[int] = Query(None, ge=0, le=1, description="Filter by working day (0:no, 1:yes)"),
    filter_hour: Optional[int] = Query(None, ge=0, le=23, description="Filter by hour (0-23)"),
    db: Session = Depends(get_db)
):
    """
    Export aggregated statistics as a downloadable CSV file.
    
    Uses the same parameters as /statistics endpoint.
    
    Parameters:
    - group_by: Group results by hour/weekday/month/season/weather/holiday/workingday
    - metric: Aggregation function (avg/sum/min/max/count)
    - start_date: Filter from this date (optional)
    - end_date: Filter until this date (optional)
    - filter_season: Filter by season 1-4 (optional)
    - filter_weather: Filter by weather 1-4 (optional)
    - filter_holiday: Filter by holiday 0/1 (optional)
    - filter_workingday: Filter by working day 0/1 (optional)
    - filter_hour: Filter by hour 0-23 (optional)
    
    Returns downloadable CSV file with aggregated statistics.
    """
    try:
        # Get statistics using shared function
        stats = calculate_statistics(
            db=db,
            group_by=group_by,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            filter_season=filter_season,
            filter_weather=filter_weather,
            filter_holiday=filter_holiday,
            filter_workingday=filter_workingday,
            filter_hour=filter_hour
        )
        
        # Convert results to DataFrame
        df = pd.DataFrame(stats["results"])
        
        # Convert to CSV
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        csv_data = stream.getvalue()
        
        # Build descriptive filename
        filename = f"statistics_{group_by}_{metric}"
        if start_date or end_date:
            filename += f"_{start_date or 'start'}_to_{end_date or 'end'}"
        filename += ".csv"
        
        # Return as streaming response
        response = StreamingResponse(
            iter([csv_data]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Export error: {str(e)}"
        )

# ==================== PREDICTION ENDPOINT ====================

@app.post("/predict")
def predict_rentals(
    season: int = Query(..., ge=1, le=4, description="Season (1:spring, 2:summer, 3:fall, 4:winter)"),
    hr: int = Query(..., ge=0, le=23, description="Hour of day (0-23)"),
    weekday: int = Query(..., ge=0, le=6, description="Day of week (0:Sunday - 6:Saturday)"),
    workingday: int = Query(..., ge=0, le=1, description="Is working day? (0:no, 1:yes)"),
    weathersit: int = Query(..., ge=1, le=4, description="Weather (1:clear, 2:mist, 3:light rain/snow, 4:heavy rain/snow)")
):
    """
    Predict bike rental count
    
    Uses a trained Random Forest model with 5 parameters.
    
    Parameters:
    - season: 1=Spring, 2=Summer, 3=Fall, 4=Winter
    - hr: Hour of the day (0-23)
    - weekday: Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
    - workingday: Is it a working day? (0=No, 1=Yes)
    - weathersit: Weather condition (1=Clear, 2=Mist, 3=Light Rain/Snow, 4=Heavy Rain/Snow)
    
    Returns predicted number of bike rentals.
    """
    try:
        # Check if model is loaded
        if predictor.model is None:
            raise HTTPException(
                status_code=503,
                detail="ML model not loaded. Please train the model using: python scripts/train_model.py"
            )
        
        # Prepare features dictionary (must match training features order)
        features: dict[str, int | float] = {
            'season': season,
            'hr': hr,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weathersit
        }
        
        # Make prediction
        prediction = predictor.predict(features)
        
        # Human-readable labels
        season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
        weather_names = {
            1: "Clear/Partly Cloudy",
            2: "Mist/Cloudy",
            3: "Light Snow/Rain",
            4: "Heavy Rain/Snow"
        }
        weekday_names = {
            0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
            4: "Thursday", 5: "Friday", 6: "Saturday"
        }
        
        return {
            "predicted_rentals": prediction,
            "input_features": features,
            "context": {
                "season": season_names.get(season, "Unknown"),
                "hour": f"{hr}:00",
                "weekday": weekday_names.get(weekday, "Unknown"),
                "weather": weather_names.get(weathersit, "Unknown"),
                "is_workingday": bool(workingday)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ==================== COMPLEX PREDICTION ENDPOINT ====================

@app.post("/predict_complex")
def predict_rentals_complex(
    season: int = Query(..., ge=1, le=4, description="Season (1:spring, 2:summer, 3:fall, 4:winter)"),
    yr: int = Query(..., ge=0, le=1, description="Year (0:2011, 1:2012)"),
    mnth: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    hr: int = Query(..., ge=0, le=23, description="Hour of day (0-23)"),
    holiday: int = Query(..., ge=0, le=1, description="Is holiday? (0:no, 1:yes)"),
    weekday: int = Query(..., ge=0, le=6, description="Day of week (0:Sunday - 6:Saturday)"),
    workingday: int = Query(..., ge=0, le=1, description="Is working day? (0:no, 1:yes)"),
    weathersit: int = Query(..., ge=1, le=4, description="Weather (1:clear, 2:mist, 3:light rain/snow, 4:heavy rain/snow)"),
    temp: float = Query(..., ge=0, le=1, description="Normalized temperature (0-1, where 1 = 41°C)"),
    atemp: float = Query(..., ge=0, le=1, description="Normalized feeling temperature (0-1, where 1 = 50°C)"),
    hum: float = Query(..., ge=0, le=1, description="Normalized humidity (0-1, where 1 = 100%)"),
    windspeed: float = Query(..., ge=0, le=1, description="Normalized wind speed (0-1, where 1 = 67 km/h)")
):
    """
    Predict bike rental count using the FULL model with all 12 features.
    
    This model is more accurate but requires more input parameters.
    Use /predict for a simpler version with only 5 parameters.
    
    Parameters:
    - season: 1=Spring, 2=Summer, 3=Fall, 4=Winter
    - yr: Year (0=2011, 1=2012)
    - mnth: Month (1-12)
    - hr: Hour of the day (0-23)
    - holiday: Is it a holiday? (0=No, 1=Yes)
    - weekday: Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
    - workingday: Is it a working day? (0=No, 1=Yes)
    - weathersit: Weather condition (1=Clear, 2=Mist, 3=Light Rain/Snow, 4=Heavy Rain/Snow)
    - temp: Normalized temperature (0-1)
    - atemp: Normalized feeling temperature (0-1)
    - hum: Normalized humidity (0-1)
    - windspeed: Normalized wind speed (0-1)
    
    Returns predicted number of bike rentals.
    """
    try:
        # Check if model is loaded
        if predictor_complex.model is None:
            raise HTTPException(
                status_code=503,
                detail="Full ML model not loaded. Please train using: python scripts/train_model_full.py"
            )
        
        # Prepare features dictionary (must match training features order)
        features = {
            'season': season,
            'yr': yr,
            'mnth': mnth,
            'hr': hr,
            'holiday': holiday,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weathersit,
            'temp': temp,
            'atemp': atemp,
            'hum': hum,
            'windspeed': windspeed
        }
        
        # Make prediction
        prediction = predictor_complex.predict(features)
        
        # Human-readable labels
        season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
        weather_names = {
            1: "Clear/Partly Cloudy",
            2: "Mist/Cloudy",
            3: "Light Snow/Rain",
            4: "Heavy Rain/Snow"
        }
        weekday_names = {
            0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
            4: "Thursday", 5: "Friday", 6: "Saturday"
        }
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        
        return {
            "predicted_rentals": prediction,
            "model": "full (12 features)",
            "input_features": features,
            "context": {
                "season": season_names.get(season, "Unknown"),
                "year": 2011 + yr,
                "month": month_names.get(mnth, "Unknown"),
                "hour": f"{hr}:00",
                "weekday": weekday_names.get(weekday, "Unknown"),
                "weather": weather_names.get(weathersit, "Unknown"),
                "is_holiday": bool(holiday),
                "is_workingday": bool(workingday),
                "temperature": f"{temp * 41:.1f}°C",
                "feels_like": f"{atemp * 50:.1f}°C",
                "humidity": f"{hum * 100:.0f}%",
                "wind_speed": f"{windspeed * 67:.1f} km/h"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ==================== HEALTH CHECK ENDPOINT ====================

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """
    Check the health status of the API.
    
    Returns status of database connection and ML model.
    """
    health_status = {
        "status": "healthy",
        "components": {}
    }
    
    # Check database connection
    try:
        db.execute(text("SELECT 1"))
        record_count = db.query(BikeRental).count()
        health_status["components"]["database"] = {
            "status": "connected",
            "records": record_count
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["database"] = {
            "status": "disconnected",
            "error": str(e)
        }
    
    # Check ML model
    if predictor.model is not None:
        health_status["components"]["ml_model"] = {
            "status": "loaded",
            "features": predictor.feature_names
        }
    else:
        health_status["components"]["ml_model"] = {
            "status": "not_loaded",
            "message": "Model needs to be trained and saved in models/ directory"
        }
    
    return health_status