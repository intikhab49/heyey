#!/usr/bin/env python3
"""
Script to fix issues with the 24h timeframe predictions
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import json

# Setup custom logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the project
from controllers.prediction import get_model, predict_next_price, prepare_features
from controllers.data_fetcher import DataFetcher
from ml_models.bilstm_predictor import BiLSTMWithAttention
import simple_config

# Constants
MODEL_PATH = simple_config.MODEL_PATH
SYMBOL = "BTC"
TIMEFRAME = "24h"
LOOKBACK = simple_config.TIMEFRAME_MAP.get(TIMEFRAME, {}).get("lookback", 72)


async def diagnose_model():
    """Diagnose issues with the 24h model"""
    logger.info(f"===== Diagnosing {TIMEFRAME} model for {SYMBOL} =====")
    
    # First, let's get the model and associated data
    try:
        model, feature_scaler, target_scaler, feature_names, training_metrics = await get_model(SYMBOL, TIMEFRAME)
        logger.info(f"Successfully loaded model for {SYMBOL} {TIMEFRAME}")
        
        # Print model's architecture
        logger.info(f"Model architecture: input_size={model.input_size}, hidden_size={model.hidden_size}, "
                    f"num_layers={model.num_layers}, dropout={model.dropout_rate}")
        
        # Print model performance metrics
        logger.info(f"Model metrics: {training_metrics}")
        r2 = training_metrics.get("val_r2", "N/A")
        mape = training_metrics.get("val_mape", "N/A")
        rmse = training_metrics.get("val_rmse", "N/A")
        logger.info(f"R²: {r2}, MAPE: {mape}%, RMSE: {rmse}")
        
        # Analyzing target scaler
        if target_scaler:
            logger.info(f"Target scaling - center: {target_scaler.mean_[0]}, scale: {target_scaler.scale_[0]}")
            logger.info(f"Target scaling range: {target_scaler.mean_[0] - 3*target_scaler.scale_[0]} to {target_scaler.mean_[0] + 3*target_scaler.scale_[0]}")
        
        # Get training data for analysis
        data_fetcher = DataFetcher()
        df = await data_fetcher.get_merged_data(SYMBOL, TIMEFRAME)
        logger.info(f"Data shape: {df.shape}")
        
        # Analyze the history of predictions
        features_df, target_series = await prepare_features(SYMBOL, TIMEFRAME)
        logger.info(f"Features shape: {features_df.shape}, Target shape: {target_series.shape}")
        
        return {
            "model": model,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_names": feature_names,
            "data": df,
            "features_df": features_df,
            "target_series": target_series,
            "training_metrics": training_metrics
        }
    except Exception as e:
        logger.error(f"Error diagnosing model: {str(e)}")
        raise


async def validate_predictions():
    """Validate predictions against actual values"""
    try:
        # Get data for analysis
        features_df, target_series = await prepare_features(SYMBOL, TIMEFRAME)
        
        # Get the model and scalers
        model, feature_scaler, target_scaler, feature_names, _ = await get_model(SYMBOL, TIMEFRAME)
        
        # Last LOOKBACK+10 days of data to use for validation
        validation_size = LOOKBACK + 10
        if len(features_df) <= validation_size:
            validation_size = len(features_df) // 2
            
        validation_features = features_df[-validation_size:].copy()
        validation_targets = target_series[-validation_size:].copy()
        
        # Create sequences for validation
        X_sequences = []
        actual_values = []
        
        for i in range(len(validation_features) - LOOKBACK):
            # Get a sequence of LOOKBACK values
            X_seq = validation_features.iloc[i:i+LOOKBACK][feature_names].values
            # Scale the features
            X_seq_scaled = feature_scaler.transform(X_seq)
            X_sequences.append(X_seq_scaled)
            
            # Store the actual next value
            actual_values.append(validation_targets.iloc[i+LOOKBACK])
        
        X_sequences = np.array(X_sequences)
        actual_values = np.array(actual_values)
        
        # Convert to tensor and predict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        predictions = []
        with torch.no_grad():
            for X_seq in X_sequences:
                X_tensor = torch.FloatTensor(X_seq.reshape(1, LOOKBACK, -1)).to(device)
                prediction = model(X_tensor)
                prediction = prediction.cpu().numpy()
                # Denormalize prediction
                prediction_denorm = target_scaler.inverse_transform(prediction.reshape(-1, 1))
                predictions.append(prediction_denorm[0, 0])
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predictions)
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        
        logger.info("======= Prediction Validation =======")
        logger.info(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        logger.info(f"Target range: {actual_values.min():.2f} to {actual_values.max():.2f}")
        logger.info(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
        
        # Calculate prediction vs actual price direction accuracy
        correct_directions = 0
        for i in range(1, len(predictions)):
            actual_direction = actual_values[i] > actual_values[i-1]
            predicted_direction = predictions[i] > predictions[i-1]
            if actual_direction == predicted_direction:
                correct_directions += 1
        
        direction_accuracy = (correct_directions / (len(predictions) - 1)) * 100
        logger.info(f"Price direction accuracy: {direction_accuracy:.2f}%")
        
        return {
            "actual_values": actual_values,
            "predictions": predictions,
            "metrics": {
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "direction_accuracy": direction_accuracy
            }
        }
    except Exception as e:
        logger.error(f"Error validating predictions: {str(e)}")
        raise


async def fix_model():
    """Fix the 24h model to improve performance"""
    try:
        logger.info(f"===== Fixing {TIMEFRAME} model for {SYMBOL} =====")
        
        # Backup the current model
        model_dir = os.path.join(MODEL_PATH, f"{SYMBOL}_{TIMEFRAME}")
        backup_dir = f"{model_dir}_backup_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
        if os.path.exists(model_dir):
            import shutil
            logger.info(f"Backing up model to {backup_dir}")
            shutil.copytree(model_dir, backup_dir)
        
        # Strategy 1: Force retrain with different hyperparameters
        logger.info("Strategy 1: Force retrain with optimized hyperparameters")
        
        # Modify the hyperparameters for 24h timeframe
        original_config = simple_config.TIMEFRAME_MAP[TIMEFRAME].copy()
        logger.info(f"Original config: {original_config}")
        
        # Use optimal hyperparameters based on performance for other timeframes
        simple_config.TIMEFRAME_MAP[TIMEFRAME].update({
            "hidden_size": 192,  # Increased from default 128
            "num_layers": 2,
            "dropout": 0.3,
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_lr": 2e-4,
            "early_stopping_patience": 30,
            "lookback": 60  # Reduced lookback to focus on more recent data
        })
        logger.info(f"Updated config: {simple_config.TIMEFRAME_MAP[TIMEFRAME]}")
        
        # Force retrain with new hyperparameters
        prediction = await predict_next_price(SYMBOL, TIMEFRAME, force_retrain=True)
        
        # Get the performance metrics after retraining
        new_metrics = prediction.get("model_performance", {})
        logger.info(f"New model metrics: {new_metrics}")
        
        return {
            "original_config": original_config,
            "new_config": simple_config.TIMEFRAME_MAP[TIMEFRAME],
            "new_performance": new_metrics
        }
    except Exception as e:
        logger.error(f"Error fixing model: {str(e)}")
        # Restore original config
        if 'original_config' in locals():
            simple_config.TIMEFRAME_MAP[TIMEFRAME] = original_config
        raise


async def compare_with_other_timeframes():
    """Compare 24h model with other timeframes"""
    timeframes = ["1h", "4h", "24h"]
    results = {}
    
    for tf in timeframes:
        logger.info(f"===== Analyzing {tf} model for {SYMBOL} =====")
        try:
            # Get prediction and model performance
            prediction = await predict_next_price(SYMBOL, tf)
            
            # Extract relevant data
            performance = prediction.get("model_performance", {})
            price_levels = prediction.get("price_levels", [])
            predicted_price = prediction.get("predicted_price")
            last_price = prediction.get("last_price")
            
            logger.info(f"{tf} model performance: R²={performance.get('r2_score')}, "
                         f"MAPE={performance.get('mape')}%, RMSE={performance.get('rmse')}")
            logger.info(f"Prediction: {predicted_price:.2f} (Last: {last_price:.2f}, "
                         f"Diff: {100 * (predicted_price - last_price) / last_price:.2f}%)")
            
            results[tf] = {
                "performance": performance,
                "predicted_price": predicted_price,
                "last_price": last_price,
                "price_levels": price_levels
            }
        except Exception as e:
            logger.error(f"Error analyzing {tf} model: {str(e)}")
            results[tf] = {"error": str(e)}
    
    return results


async def main():
    """Main entry point"""
    try:
        # First, diagnose the current model
        logger.info("Step 1: Diagnosing current model")
        diagnosis = await diagnose_model()
        
        # Validate predictions
        logger.info("Step 2: Validating predictions")
        validation = await validate_predictions()
        
        # Compare with other timeframes
        logger.info("Step 3: Comparing with other timeframes")
        comparison = await compare_with_other_timeframes()
        
        # Fix the model
        logger.info("Step 4: Fixing the model")
        fix_results = await fix_model()
        
        # Validate after fixing
        logger.info("Step 5: Validating after fixing")
        post_fix_validation = await validate_predictions()
        
        # Print summary
        logger.info("\n===== SUMMARY =====")
        logger.info("Before fixing:")
        logger.info(f"R²: {validation['metrics']['r2']:.4f}, MAPE: {validation['metrics']['mape']:.2f}%, RMSE: {validation['metrics']['rmse']:.2f}")
        
        logger.info("After fixing:")
        logger.info(f"R²: {post_fix_validation['metrics']['r2']:.4f}, MAPE: {post_fix_validation['metrics']['mape']:.2f}%, RMSE: {post_fix_validation['metrics']['rmse']:.2f}")
        
        logger.info("Improvement:")
        r2_improvement = post_fix_validation['metrics']['r2'] - validation['metrics']['r2']
        mape_improvement = validation['metrics']['mape'] - post_fix_validation['metrics']['mape']
        rmse_improvement = validation['metrics']['rmse'] - post_fix_validation['metrics']['rmse']
        
        logger.info(f"R² improvement: {r2_improvement:.4f} ({r2_improvement / abs(validation['metrics']['r2'] + 1e-8) * 100:.2f}%)")
        logger.info(f"MAPE improvement: {mape_improvement:.2f}% ({mape_improvement / validation['metrics']['mape'] * 100:.2f}%)")
        logger.info(f"RMSE improvement: {rmse_improvement:.2f} ({rmse_improvement / validation['metrics']['rmse'] * 100:.2f}%)")
        
        return {
            "diagnosis": diagnosis,
            "before": validation['metrics'],
            "after": post_fix_validation['metrics'],
            "improvement": {
                "r2": r2_improvement,
                "mape": mape_improvement,
                "rmse": rmse_improvement
            }
        }
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
