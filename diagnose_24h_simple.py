#!/usr/bin/env python
"""
Simplified script to diagnose issues with the 24h prediction model.
"""
import os
import sys
import logging
import asyncio
import json
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the script can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from controllers.prediction import _get_and_prepare_training_data, get_model, predict_next_price
    from controllers.data_fetcher import DataFetcher
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

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
        
        # Print feature names
        print(f"\nFeatures used for model:")
        for i, feature in enumerate(feature_names):
            print(f"{i+1}. {feature}")
        
        # Print target scaling info
        print("\nTarget scaling info:")
        try:
            print(f"Mean: {target_scaler.mean_[0]}")
            print(f"Scale: {target_scaler.scale_[0]}")
        except:
            print("Could not access target scaler parameters")
        
        # Print sample of unscaled target values
        try:
            y_unscaled = target_scaler.inverse_transform(y_train.cpu().numpy())
            print("\nSample of unscaled target values:")
            print(f"First 5: {y_unscaled[:5].flatten()}")
            print(f"Min: {y_unscaled.min()}")
            print(f"Max: {y_unscaled.max()}")
            print(f"Mean: {y_unscaled.mean()}")
        except Exception as e:
            print(f"Error getting unscaled values: {e}")
            
    except Exception as e:
        logger.error(f"Error analyzing model data: {str(e)}")

async def analyze_predictions():
    """Analyze recent predictions for various timeframes"""
    symbol = "BTC"
    timeframes = ["1h", "4h", "24h"]
    
    for timeframe in timeframes:
        logger.info(f"Getting prediction for {symbol} {timeframe}...")
        try:
            prediction = await predict_next_price(symbol, timeframe)
            
            print(f"\n{timeframe} Prediction Results:")
            print(f"Latest price: {prediction['last_price']}")
            print(f"Predicted price: {prediction['predicted_price']}")
            print(f"Difference: {prediction['predicted_price'] - prediction['last_price']}")
            print(f"Pct difference: {(prediction['predicted_price'] - prediction['last_price']) / prediction['last_price'] * 100:.2f}%")
            
            # Print model metrics
            metrics = prediction.get('model_performance', {})
            print(f"\nModel Performance:")
            print(f"R² Score: {metrics.get('r2_score', 'N/A')}")
            print(f"MAPE: {metrics.get('mape', 'N/A')}")
            print(f"RMSE: {metrics.get('rmse', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error getting prediction for {timeframe}: {str(e)}")

async def main():
    """Main function to run the diagnosis"""
    print("\n====== 24h Model Diagnosis ======\n")
    
    # Part 1: Analyze model training data
    await analyze_model_data()
    
    # Part 2: Analyze recent predictions for different timeframes
    await analyze_predictions()
    
    print("\n====== Diagnosis Complete ======")

if __name__ == "__main__":
    asyncio.run(main())
