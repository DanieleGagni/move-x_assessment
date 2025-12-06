"""
Machine Learning model module for bike rental predictions.

This module handles loading the trained model and performing inference
The model is trained offline using scripts/train_model.py and loaded at runtime.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BikeRentalPredictor:
    """
    Predictor class for bike rental count prediction.
    
    Loads a pre-trained model and provides methods for making predictions.
    """
    
    # Features used for prediction (must match training features in train_model.py)
    FEATURE_NAMES: List[str] = [
        'season', 'hr', 'weekday', 'workingday', 'weathersit'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file. 
                       Defaults to 'models/bike_rental_model.joblib'
        """
        self.model = None
        self.model_path = Path(model_path) if model_path else Path("models/bike_rental_model.joblib")
        self.feature_names = self.FEATURE_NAMES
        
        # Try to load model on initialization
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                logger.warning(
                    f"Model file not found at {self.model_path}. "
                    "Please train the model using scripts/train_model.py"
                )
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def reload_model(self) -> bool:
        """
        Reload the model from disk.
        
        Useful after retraining the model.
        
        Returns:
            True if model reloaded successfully, False otherwise.
        """
        return self._load_model()
    
    def predict(self, features: Dict[str, Union[int, float]]) -> int:
        """
        Make a prediction for bike rental count.
        
        Args:
            features: Dictionary containing feature values.
                     Must include all features in FEATURE_NAMES.
        
        Returns:
            Predicted bike rental count (rounded to nearest integer).
        
        Raises:
            ValueError: If model is not loaded or features are missing.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please train and save the model first.")
        
        # Validate all required features are present
        missing_features = [f for f in self.feature_names if f not in features]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Create feature array in correct order
        feature_values = [features[name] for name in self.feature_names]
        X = np.array([feature_values])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Ensure non-negative prediction
        return max(0, int(round(prediction)))
    
    def predict_batch(self, features_list: List[Dict[str, Union[int, float]]]) -> List[int]:
        """
        Make predictions for multiple samples.
        
        Args:
            features_list: List of feature dictionaries.
        
        Returns:
            List of predicted bike rental counts.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please train and save the model first.")
        
        # Create feature matrix
        X = np.array([
            [features[name] for name in self.feature_names]
            for features in features_list
        ])
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        return [max(0, int(round(p))) for p in predictions]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the model (if available).
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance.
        """
        if self.model is None:
            return None
        
        try:
            # Try to get feature importance (works for tree-based models)
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return dict(zip(self.feature_names, importance))
            # Try coefficients (works for linear models)
            elif hasattr(self.model, 'coef_'):
                coef = self.model.coef_
                return dict(zip(self.feature_names, coef))
        except Exception:
            pass
        
        return None


# ==================== SIMPLE MODEL (5 features) ====================
# Global predictor instance for the simple model
# This is loaded once when the module is imported
predictor = BikeRentalPredictor()


# ==================== FULL MODEL (12 features) ====================
# Full feature set for maximum accuracy
FULL_FEATURE_NAMES = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
]

# Create a second predictor for the full model
predictor_complex = BikeRentalPredictor(model_path="models/bike_rental_model_full.joblib")
predictor_complex.feature_names = FULL_FEATURE_NAMES
