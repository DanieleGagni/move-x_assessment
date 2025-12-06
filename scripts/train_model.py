"""
Training script for the Bike Rental Prediction Model.

This script trains a machine learning model on the bike sharing dataset
and saves it for use by the API.

Usage:
    python scripts/train_model.py

The trained model will be saved to models/bike_rental_model.joblib
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt

# Feature columns used for training
FEATURE_COLUMNS = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed' ]


# Target column
TARGET_COLUMN = 'cnt'


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the bike sharing dataset from CSV.
    
    Args:
        data_path: Path to the CSV file.
    
    Returns:
        DataFrame containing the dataset.
    """
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df['dteday'].min()} to {df['dteday'].max()}")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target vector.
    
    Args:
        df: Raw DataFrame.
    
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target vector.
    """
    # Check for missing columns
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    
    return X, y


def get_model(model_type: str):
    """
    Get a model instance based on type.
    
    Args:
        model_type: Type of model ('random_forest', 'gradient_boosting', 
                   'decision_tree', 'linear', 'ridge')
    
    Returns:
        Scikit-learn model instance.
    """
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=15,          
            min_samples_split=7,   
            min_samples_leaf=3,    
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        ),
        'decision_tree': DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0)
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(models.keys())}")
    
    return models[model_type]


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'random_forest'):
    """
    Train the model and evaluate performance.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        model_type: Type of model to train.
    
    Returns:
        Trained model instance.
    """
    logger.info(f"Training {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Get and train model
    model = get_model(model_type)
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    logger.info("Training Set Performance:")
    logger.info(f"  RMSE: {train_rmse:.2f}")
    logger.info(f"  MAE: {train_mae:.2f}")
    logger.info(f"  R²: {train_r2:.4f}")
    
    # Evaluate on test set
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    logger.info("Test Set Performance:")
    logger.info(f"  RMSE: {test_rmse:.2f}")
    logger.info(f"  MAE: {test_mae:.2f}")
    logger.info(f"  R²: {test_r2:.4f}")
    
    # Cross-validation
    logger.info("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    logger.info(f"  CV R² scores: {cv_scores}")
    logger.info(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        logger.info("Feature Importances:")
        importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feat}: {imp:.4f}")
    
    return model


def save_model(model, output_path: str):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model instance.
        output_path: Path to save the model.
    """
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    
    # Command Line arguments
    parser = argparse.ArgumentParser(description='Train bike rental prediction model')
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/hour.csv',
        help='Path to the training data CSV file'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'decision_tree', 'linear', 'ridge'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models/bike_rental_model.joblib',
        help='Path to save the trained model'
    )
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.info("Please ensure the dataset is in the correct location.")
        sys.exit(1)
    
    try:
        # Load and prepare data
        df = load_data(args.data)
        X, y = prepare_features(df)
        
        # Train model
        model = train_model(X, y, model_type=args.model)
        
        # Save model
        save_model(model, args.output)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
