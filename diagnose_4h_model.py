#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to diagnose issues with the 4h timeframe model
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.prediction import get_model, prepare_features, predict_next_price
from controllers.data_fetcher import DataFetcher

async def diagnose_4h_model(symbol="BTC"):
    """
    Diagnose issues with the 4h timeframe model for a given symbol
    """
    timeframe = "4h"
    logger.info(f"Starting diagnosis of {timeframe} model for {symbol}")
    
    try:
        # First, get historical data for the symbol
        logger.info("Fetching historical data...")
        data_fetcher = DataFetcher()
        historical_data = await data_fetcher.get_merged_data(symbol, timeframe)
        
        if historical_data is None or historical_data.empty:
            logger.error(f"Failed to fetch data for {symbol} {timeframe}")
            return
            
        logger.info(f"Fetched {len(historical_data)} data points")
        logger.info(f"Data range: {historical_data.index.min()} to {historical_data.index.max()}")
        logger.info(f"Price range: {historical_data['Close'].min()} to {historical_data['Close'].max()}")
        logger.info(f"Last 5 prices: {historical_data['Close'].tail()}")
        
        # Now, prepare features and target for training/evaluation
        features_df, target_series = await prepare_features(symbol, timeframe)
        logger.info(f"Prepared features shape: {features_df.shape}")
        logger.info(f"Target shape: {len(target_series)}")
        logger.info(f"Target type: absolute price (for 4h)")
        logger.info(f"Target range: {target_series.min()} to {target_series.max()}")
        logger.info(f"Last 5 target values: {target_series.tail()}")
        
        # Feature analysis
        logger.info(f"Feature columns: {features_df.columns.tolist()}")
        numerical_features = features_df.select_dtypes(include=['number'])
        for col in numerical_features.columns:
            if numerical_features[col].min() >= -10 and numerical_features[col].max() <= 10:
                logger.info(f"Feature {col} appears to be normalized: [{numerical_features[col].min()}, {numerical_features[col].max()}]")
            else:
                logger.info(f"Feature {col} appears to be unnormalized: [{numerical_features[col].min()}, {numerical_features[col].max()}]")
        
        # Get model and make predictions
        logger.info("Loading model...")
        model, feature_scaler, target_scaler, feature_names, training_metrics = await get_model(symbol, timeframe)
        
        logger.info(f"Model performance metrics: {training_metrics}")
        
        # Get hyperparameters for this symbol/timeframe
        from simple_config import TIMEFRAME_MAP
        timeframe_config = TIMEFRAME_MAP.get(timeframe, {})
        lookback = timeframe_config.get('lookback', 60)
        
        # Prepare test data (last 30 days)
        test_end = historical_data.index.max()
        test_start = test_end - timedelta(days=30)
        test_data = historical_data.loc[test_start:test_end]
        logger.info(f"Using test data from {test_start} to {test_end} ({len(test_data)} points)")
        
        # Create predictions for the last 30 days (by sliding window)
        predictions = []
        actuals = []
        
        # We need at least lookback+1 points to make a prediction
        min_required = lookback + 1
        if len(test_data) <= min_required:
            logger.warning(f"Not enough test data for evaluation: {len(test_data)} <= {min_required}")
            test_data = historical_data.iloc[-min_required*2:]  # Take more data if needed
            logger.info(f"Using extended test data with {len(test_data)} points")
        
        # For each test day, make a prediction using data available up to that point
        logger.info("Making predictions on historical data...")
        model.eval()  # Ensure model is in evaluation mode
        
        for i in range(lookback, len(test_data)-1):
            current_data = test_data.iloc[:i+1]  # Data up to current 4h
            next_actual = test_data.iloc[i+1]['Close']  # Next 4h's close price
            
            # Prepare features
            curr_features = features_df.loc[current_data.index]
            if len(curr_features) < lookback:
                logger.warning(f"Insufficient features data at index {i}: {len(curr_features)} < {lookback}")
                continue
                
            # Get the last lookback features
            X = curr_features[-lookback:][feature_names].values
            X_scaled = feature_scaler.transform(X)
            X_seq = X_scaled.reshape(1, lookback, -1)
            
            # Convert to tensor
            import torch
            X_tensor = torch.FloatTensor(X_seq)
            
            # Make prediction
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
                
            # Inverse transform prediction
            pred_value = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            
            # Store results
            predictions.append(pred_value)
            actuals.append(next_actual)
            logger.info(f"Time: {test_data.index[i+1]}, Predicted: {pred_value:.2f}, Actual: {next_actual:.2f}, Diff: {pred_value - next_actual:.2f} ({100*(pred_value-next_actual)/next_actual:.2f}%)")
        
        # Calculate performance metrics
        if predictions and actuals:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions)
            mape = mean_absolute_percentage_error(actuals, predictions) * 100
            
            logger.info("====== Performance Metrics ======")
            logger.info(f"RMSE: {rmse:.2f}")
            logger.info(f"R2 Score: {r2:.4f}")
            logger.info(f"MAPE: {mape:.2f}%")
            
            # Check for large prediction errors
            abs_errors = np.abs(predictions - actuals)
            pct_errors = abs_errors / actuals * 100
            large_errors = np.where(pct_errors > 5)[0]  # Errors > 5%
            
            if len(large_errors) > 0:
                logger.info(f"Found {len(large_errors)} large prediction errors (>5%):")
                for idx in large_errors:
                    logger.info(f"  Time: {test_data.index[lookback+idx+1]}, Predicted: {predictions[idx]:.2f}, Actual: {actuals[idx]:.2f}, Error: {pct_errors[idx]:.2f}%")
                    
            # Detailed error analysis
            logger.info("====== Error Analysis ======")
            logger.info(f"Average error: {np.mean(abs_errors):.2f} ({np.mean(pct_errors):.2f}%)")
            logger.info(f"Median error: {np.median(abs_errors):.2f} ({np.median(pct_errors):.2f}%)")
            logger.info(f"Max error: {np.max(abs_errors):.2f} ({np.max(pct_errors):.2f}%)")
            logger.info(f"Min error: {np.min(abs_errors):.2f} ({np.min(pct_errors):.2f}%)")
            
            # Check for overfit
            if r2 < -1:
                logger.warning("Model appears to be severely overfit or poorly trained (R2 < -1)")
            
            # Check for systematic bias
            bias = np.mean(predictions - actuals)
            if abs(bias) > 1000:
                logger.warning(f"Model shows significant bias: {bias:.2f}")
                
            # Target scaling diagnosis
            logger.info("====== Target Scaling Analysis ======")
            logger.info(f"Target scaler mean: {target_scaler.mean_[0]:.4f}")
            logger.info(f"Target scaler scale: {target_scaler.scale_[0]:.4f}")
            
            # If target is absolute price, check if scaling is appropriate
            target_sample = target_series.sample(min(100, len(target_series)))
            target_sample_scaled = target_scaler.transform(target_sample.values.reshape(-1, 1)).flatten()
            logger.info(f"Scaled target range: [{target_sample_scaled.min():.4f}, {target_sample_scaled.max():.4f}]")
            
            if target_sample_scaled.min() < -10 or target_sample_scaled.max() > 10:
                logger.warning("Scaled target values outside normal range [-10, 10], could indicate scaling issues")
                
        # Make a fresh prediction for the next timeframe
        logger.info("====== Fresh Prediction ======")
        prediction_result = await predict_next_price(symbol, timeframe)
        logger.info(f"Next 4h prediction: {prediction_result['predicted_price']:.2f} (last price: {prediction_result['last_price']:.2f})")
        logger.info(f"Relative change: {100 * (prediction_result['predicted_price'] - prediction_result['last_price']) / prediction_result['last_price']:.2f}%")
        
        # Test model retraining to see if it helps
        logger.info("====== Testing Model Retraining ======")
        retrained_result = await predict_next_price(symbol, timeframe, force_retrain=True)
        logger.info(f"Retrained model prediction: {retrained_result['predicted_price']:.2f}")
        logger.info(f"Retrained model metrics: {retrained_result['model_performance']}")
            
    except Exception as e:
        logger.error(f"Error during diagnosis: {str(e)}", exc_info=True)

if __name__ == "__main__":
    symbol = "BTC" if len(sys.argv) < 2 else sys.argv[1]
    asyncio.run(diagnose_4h_model(symbol))
