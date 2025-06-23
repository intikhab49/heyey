#!/usr/bin/env python3
"""
Comprehensive Prediction Accuracy Diagnosis
Identifies root causes of poor prediction performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import requests
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    """Analyze the quality and characteristics of the input data"""
    print("=== DATA QUALITY ANALYSIS ===")
    
    # Mock CoinGecko API call to get recent BTC data
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "8", "interval": "hourly"}
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Convert to DataFrame
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        print(f"Data Points: {len(df)}")
        print(f"Time Range: {df.index.min()} to {df.index.max()}")
        print(f"Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        
        # Calculate volatility metrics
        df['returns'] = df['price'].pct_change()
        df['volatility_1h'] = df['returns'].rolling(window=24).std() * np.sqrt(24)  # Annualized
        
        print(f"\n--- VOLATILITY ANALYSIS ---")
        print(f"Mean 1h return: {df['returns'].mean()*100:.4f}%")
        print(f"Std 1h return: {df['returns'].std()*100:.4f}%")
        print(f"Max 1h move: {df['returns'].abs().max()*100:.2f}%")
        print(f"24h rolling volatility: {df['volatility_1h'].mean()*100:.2f}%")
        
        # Trend analysis
        df['sma_24'] = df['price'].rolling(window=24).mean()
        df['trend'] = np.where(df['price'] > df['sma_24'], 1, -1)
        trend_changes = (df['trend'].diff() != 0).sum()
        print(f"Trend changes in 8 days: {trend_changes}")
        
        # Autocorrelation analysis
        autocorr_1 = df['returns'].autocorr(lag=1)
        autocorr_24 = df['returns'].autocorr(lag=24)
        print(f"1-hour autocorrelation: {autocorr_1:.4f}")
        print(f"24-hour autocorrelation: {autocorr_24:.4f}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def analyze_model_architecture_issues():
    """Analyze issues with the model architecture and training"""
    print("\n=== MODEL ARCHITECTURE ISSUES ===")
    
    # From the logs
    model_metrics = {
        "1h": {
            "r2": -0.9197,
            "mape": 1.43,
            "rmse": 1978.06,
            "val_loss": 0.475,
            "lookback": 48,
            "hidden_size": 64,
            "input_features": 34,
            "training_samples": 115,
            "validation_samples": 29
        }
    }
    
    print("--- 1H MODEL ANALYSIS ---")
    metrics = model_metrics["1h"]
    
    # R² analysis
    print(f"R² Score: {metrics['r2']:.4f}")
    if metrics['r2'] < 0:
        print("❌ CRITICAL: Negative R² indicates model performs worse than predicting the mean")
        print("   This suggests severe overfitting or inappropriate model for the data")
    
    # Sample size analysis
    total_samples = metrics['training_samples'] + metrics['validation_samples']
    lookback_ratio = total_samples / metrics['lookback']
    print(f"\nSample Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Lookback: {metrics['lookback']} hours")
    print(f"Sample-to-lookback ratio: {lookback_ratio:.2f}")
    
    if lookback_ratio < 10:
        print("❌ CRITICAL: Very low sample-to-lookback ratio")
        print("   Need at least 10x more samples than lookback period")
    
    # Feature-to-sample ratio
    feature_ratio = metrics['input_features'] / metrics['training_samples']
    print(f"Feature-to-sample ratio: {feature_ratio:.2f}")
    if feature_ratio > 0.1:
        print("❌ WARNING: High feature-to-sample ratio suggests overfitting risk")
    
    # MAPE vs Price analysis
    btc_price = 99415.92  # From logs
    rmse_percentage = (metrics['rmse'] / btc_price) * 100
    print(f"\nError Analysis:")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"RMSE as % of price: {rmse_percentage:.2f}%")

def analyze_prediction_logic_issues():
    """Analyze issues in the prediction logic and probability calculation"""
    print("\n=== PREDICTION LOGIC ISSUES ===")
    
    # From the prediction output
    prediction_data = {
        "current_price": 99415.92,
        "predicted_price": 103308.32,
        "probability": 0.666,
        "model_confidence": 0.8,
        "r2": -0.9197,
        "mape": 1.43
    }
    
    price_diff_pct = ((prediction_data["predicted_price"] - prediction_data["current_price"]) / 
                     prediction_data["current_price"]) * 100
    
    print(f"Predicted price change: +{price_diff_pct:.2f}%")
    print(f"Model R²: {prediction_data['r2']:.4f}")
    print(f"Assigned probability: {prediction_data['probability']:.3f}")
    print(f"Model confidence: {prediction_data['model_confidence']:.3f}")
    
    # Logical inconsistencies
    print("\n--- LOGICAL INCONSISTENCIES ---")
    
    if prediction_data["r2"] < 0 and prediction_data["model_confidence"] > 0.5:
        print("❌ CRITICAL: High model confidence despite negative R²")
        print("   The system should lower confidence for poor-performing models")
    
    if prediction_data["probability"] > 0.5 and prediction_data["r2"] < -0.5:
        print("❌ CRITICAL: High prediction probability despite very poor model performance")
        print("   Probability calculation should account for model quality")
    
    # Technical indicator conflicts
    print("\n--- TECHNICAL INDICATOR ANALYSIS ---")
    print("RSI: oversold (bullish signal)")
    print("MACD: bearish_trend (bearish signal)")
    print("Bollinger: above_upper + continuation (mixed signal)")
    print("Volume: high (confirms move)")
    print("❌ WARNING: Conflicting technical signals not properly weighted")

def analyze_training_process_issues():
    """Analyze issues in the training process"""
    print("\n=== TRAINING PROCESS ISSUES ===")
    
    # From training logs
    training_metrics = {
        "initial_val_loss": 2.7864,
        "final_val_loss": 0.4750,
        "best_epoch": 45,
        "total_epochs": 70,
        "early_stopping_patience": 25,
        "final_r2": -0.9197,
        "improvement_pattern": "steady_decrease_then_plateau"
    }
    
    print(f"Training stopped at epoch {training_metrics['total_epochs']}")
    print(f"Best model from epoch {training_metrics['best_epoch']}")
    print(f"Validation loss: {training_metrics['initial_val_loss']:.4f} → {training_metrics['final_val_loss']:.4f}")
    
    # Loss analysis
    loss_reduction = ((training_metrics['initial_val_loss'] - training_metrics['final_val_loss']) / 
                     training_metrics['initial_val_loss']) * 100
    print(f"Loss reduction: {loss_reduction:.1f}%")
    
    print("\n--- TRAINING ISSUES IDENTIFIED ---")
    
    if training_metrics['final_r2'] < 0:
        print("❌ CRITICAL: Model failed to learn meaningful patterns")
        print("   Possible causes:")
        print("   - Data is too noisy for the model architecture")
        print("   - Insufficient training data")
        print("   - Inappropriate loss function for the task")
        print("   - Features don't contain predictive information")
    
    # Data leakage check
    print("\n--- POTENTIAL DATA LEAKAGE ---")
    print("⚠️  WARNING: Using scaled features for prediction")
    print("   Ensure scaler was fit only on training data")
    print("   Check for future information leaking into features")

def recommend_fixes():
    """Provide specific recommendations to fix prediction accuracy"""
    print("\n=== RECOMMENDED FIXES ===")
    
    print("🔧 IMMEDIATE FIXES:")
    print("1. REDUCE LOOKBACK PERIOD")
    print("   - Change from 48h to 12-24h for 1h predictions")
    print("   - Crypto markets have short memory")
    
    print("\n2. INCREASE TRAINING DATA")
    print("   - Use 30+ days instead of 8 days")
    print("   - Need minimum 500+ samples for stable training")
    
    print("\n3. FIX CONFIDENCE CALCULATION")
    print("   - Set confidence = max(0.1, (R² + 1) / 2) for R² > -1")
    print("   - For R² < -1, set confidence = 0.1")
    
    print("\n4. IMPROVE FEATURE SELECTION")
    print("   - Remove highly correlated features more aggressively")
    print("   - Focus on 10-15 most predictive features")
    
    print("\n🏗️  ARCHITECTURAL CHANGES:")
    print("1. CHANGE MODEL ARCHITECTURE")
    print("   - Try simpler models (Linear/XGBoost) for comparison")
    print("   - Reduce hidden size to 32")
    print("   - Add regularization (L1/L2)")
    
    print("\n2. ENSEMBLE APPROACH")
    print("   - Combine multiple timeframe predictions")
    print("   - Weight by individual model performance")
    
    print("\n3. TARGET ENGINEERING")
    print("   - Predict price direction (up/down) instead of exact price")
    print("   - Use percentage change instead of absolute price")
    
    print("\n📊 DATA IMPROVEMENTS:")
    print("1. FEATURE ENGINEERING")
    print("   - Add market sentiment indicators")
    print("   - Include external data (news, social sentiment)")
    print("   - Use multiple timeframe features")
    
    print("\n2. DATA QUALITY")
    print("   - Remove outliers more aggressively")
    print("   - Handle market regime changes")
    print("   - Add volume-weighted features")

def main():
    """Run comprehensive prediction accuracy diagnosis"""
    print("🔍 CRYPTOAION PREDICTION ACCURACY DIAGNOSIS")
    print("=" * 60)
    
    # Analyze different aspects
    df = analyze_data_quality()
    analyze_model_architecture_issues()
    analyze_prediction_logic_issues() 
    analyze_training_process_issues()
    recommend_fixes()
    
    print("\n" + "=" * 60)
    print("💡 SUMMARY: The main issues are:")
    print("1. Model architecture inappropriate for noisy 1h data")
    print("2. Insufficient training data (8 days)")
    print("3. Poor confidence/probability calculation")
    print("4. Too many features for available samples")
    print("5. Conflicting technical indicators not properly handled")
    
    print("\n🎯 PRIORITY ACTIONS:")
    print("1. Increase training data to 30+ days")
    print("2. Reduce lookback to 12-24 hours")
    print("3. Fix confidence calculation based on R²")
    print("4. Reduce feature count to 10-15")
    print("5. Consider predicting direction instead of exact price")

if __name__ == "__main__":
    main()
