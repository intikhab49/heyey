#!/usr/bin/env python
"""
Script to diagnose issues with the 24h prediction model.
This will analyze the model, data, and predictions to identify why 24h predictions are poor.
"""
import os
import sys
import logging
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.prediction import _get_and_prepare_training_data, get_model, predict_next_price
from controllers.data_fetcher import DataFetcher

async def analyze_model_data():
    """Analyze the data used for 24h model training"""
    symbol = "BTC"
    timeframe = "24h"
    
    logger.info(f"Analyzing training data for {symbol} {timeframe}...")
    try:
        # Get training data
        X_train, y_train, X_val, y_val, feature_scaler, target_scaler, feature_names, features_df = await _get_and_prepare_training_data(
            symbol, timeframe, 72
        )
        
        # Print basic statistics
        print(f"\nTraining Data Statistics:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Print target statistics (before scaling)
        y_data_original = target_scaler.inverse_transform(y_train.numpy())
        print(f"\nTarget (y) statistics (before normalization):")
        print(f"Min: {y_data_original.min()}")
        print(f"Max: {y_data_original.max()}")
        print(f"Mean: {y_data_original.mean()}")
        print(f"Std: {y_data_original.std()}")
        
        # Check for any anomalies in the target data
        print("\nChecking for anomalies in target data...")
        y_diff = np.diff(y_data_original.flatten())
        max_change = np.max(np.abs(y_diff))
        max_pct_change = np.max(np.abs(y_diff / y_data_original[:-1].flatten())) * 100
        
        print(f"Max absolute change between consecutive targets: {max_change}")
        print(f"Max percentage change between consecutive targets: {max_pct_change:.2f}%")
        
        if max_pct_change > 20:
            print(f"WARNING: Detected large price swings in target data (>{max_pct_change:.2f}%)")
        
        # Check scaling parameters
        print("\nScaling parameters:")
        print(f"Feature scaler mean (first 5): {feature_scaler.mean_[:5]}")
        print(f"Feature scaler scale (first 5): {feature_scaler.scale_[:5]}")
        print(f"Target scaler mean: {target_scaler.mean_}")
        print(f"Target scaler scale: {target_scaler.scale_}")
        
        # Feature analysis
        print("\nFeature analysis:")
        for i, feature in enumerate(feature_names[:10]):  # Show first 10 features
            print(f"Feature {i}: {feature}")
            
        return {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_names": feature_names,
            "features_df": features_df,
            "y_data_original": y_data_original
        }
            
    except Exception as e:
        logger.error(f"Error analyzing model data: {str(e)}")
        return None

async def analyze_predictions():
    """Analyze recent predictions and their errors"""
    symbol = "BTC"
    timeframe = "24h"
    
    logger.info(f"Analyzing recent predictions for {symbol} {timeframe}...")
    try:
        # Get model and prepare it for prediction
        model, feature_scaler, target_scaler, feature_names = await get_model(symbol, timeframe)
        
        # Get predictions
        prediction = await predict_next_price(symbol, timeframe)
        
        print(f"\nPrediction Results:")
        print(f"Latest price: {prediction['last_price']}")
        print(f"Predicted price: {prediction['predicted_price']}")
        print(f"Difference: {prediction['predicted_price'] - prediction['last_price']}")
        print(f"Percentage difference: {(prediction['predicted_price'] - prediction['last_price']) / prediction['last_price'] * 100:.2f}%")
        
        # Analyze model performance metrics
        print(f"\nModel Performance Metrics:")
        performance = prediction.get('model_performance', {})
        print(f"R² Score: {performance.get('r2_score', 'N/A')}")
        print(f"MAPE: {performance.get('mape', 'N/A')}")
        print(f"RMSE: {performance.get('rmse', 'N/A')}")
        print(f"Validation Loss: {performance.get('validation_loss', 'N/A')}")
        
        # Analyze support/resistance levels
        print(f"\nSupport/Resistance Levels:")
        for level in prediction.get('price_levels', []):
            print(f"{level['type'].capitalize()}: {level['price']}, Strength: {level['strength']}")
        
        # Analyze technical indicators
        print(f"\nTechnical Indicators:")
        for indicator in prediction.get('confidence_factors', {}).get('technical_indicators', []):
            indicator_name = indicator.get('indicator', 'Unknown')
            signal = indicator.get('signal', 'Unknown')
            print(f"{indicator_name}: {signal}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error analyzing predictions: {str(e)}")
        return None

async def compare_with_other_timeframes():
    """Compare 24h predictions with other timeframes to identify differences"""
    symbol = "BTC"
    timeframes = ["1h", "4h", "24h"]
    
    logger.info(f"Comparing model predictions across timeframes...")
    try:
        results = {}
        
        for timeframe in timeframes:
            print(f"\n--- Analyzing {timeframe} timeframe ---")
            prediction = await predict_next_price(symbol, timeframe)
            
            results[timeframe] = {
                "last_price": prediction['last_price'],
                "predicted_price": prediction['predicted_price'],
                "r2_score": prediction.get('model_performance', {}).get('r2_score', 'N/A'),
                "mape": prediction.get('model_performance', {}).get('mape', 'N/A'),
                "rmse": prediction.get('model_performance', {}).get('rmse', 'N/A')
            }
            
            print(f"Latest price: {prediction['last_price']}")
            print(f"Predicted price: {prediction['predicted_price']}")
            print(f"Percentage difference: {(prediction['predicted_price'] - prediction['last_price']) / prediction['last_price'] * 100:.2f}%")
            print(f"R² Score: {results[timeframe]['r2_score']}")
            print(f"MAPE: {results[timeframe]['mape']}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error comparing timeframes: {str(e)}")
        return None

async def main():
    """Main function to run diagnosis"""
    print("\n====== 24h Model Diagnosis ======\n")
    
    # Part 1: Analyze model training data
    data_analysis = await analyze_model_data()
    
    # Part 2: Analyze recent predictions
    prediction_analysis = await analyze_predictions()
    
    # Part 3: Compare with other timeframes
    comparison = await compare_with_other_timeframes()
    
    print("\n====== Diagnosis Summary ======\n")
    if data_analysis and prediction_analysis:
        # Summarize findings
        if prediction_analysis.get('model_performance', {}).get('r2_score', 0) < -1:
            print("ISSUE DETECTED: Negative R² score indicates the model is performing worse than a simple mean predictor.")
            print("This suggests serious problems with the model or the training data.")
        
        # Compare target scaling
        y_std = float(data_analysis["y_data_original"].std())
        target_scale = float(data_analysis["target_scaler"].scale_)
        
        print(f"\nTarget scaling analysis:")
        print(f"Original target standard deviation: {y_std}")
        print(f"Target scaler scale parameter: {target_scale}")
        
        if abs(1 - y_std / target_scale) > 0.1:
            print("ISSUE DETECTED: Target scaling may be inconsistent with the data distribution.")
        
        print("\nPossible issues based on diagnosis:")
        
        # Analyze results and suggest fixes
        if prediction_analysis.get('model_performance', {}).get('r2_score', 0) < -1:
            print("1. The model is severely underfit or overfit.")
            print("2. The target variable might be too noisy or unpredictable.")
            print("3. There might be data leakage or preprocessing issues.")
            
        print("\nRecommended actions:")
        print("1. Review the training data quality and preprocessing.")
        print("2. Consider changing the target variable (absolute vs. percentage change).")
        print("3. Try a different model architecture or hyperparameters.")
        print("4. Increase the history length or add more relevant features.")
    
    else:
        print("Diagnosis couldn't be completed due to errors. Check the logs.")

if __name__ == "__main__":
    asyncio.run(main())
