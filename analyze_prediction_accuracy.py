#!/usr/bin/env python3
"""
Analyze prediction accuracy issues
"""

import sys
import os
import json
import requests
from datetime import datetime, timedelta

def check_prediction_accuracy():
    """Check current prediction accuracy issues"""
    
    print("=== CryptoAion Prediction Accuracy Analysis ===\n")
    
    # Test different timeframes
    timeframes = ['1h', '4h', '24h']
    base_url = "http://localhost:8000"
    
    for tf in timeframes:
        print(f"\n--- {tf} Timeframe Analysis ---")
        
        try:
            # Get prediction
            response = requests.get(f"{base_url}/api/predict/BTC?timeframe={tf}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key metrics
                predicted = data.get('predicted_price', 0)
                current = data.get('last_price', 0)
                
                # Model performance
                perf = data.get('model_performance', {})
                r2 = perf.get('r2_score', 'N/A')
                mape = perf.get('mape', 'N/A')
                rmse = perf.get('rmse', 'N/A')
                val_loss = perf.get('validation_loss', 'N/A')
                
                # Prediction vs current price
                if current > 0:
                    price_diff_pct = ((predicted - current) / current) * 100
                else:
                    price_diff_pct = 0
                
                print(f"  Current Price: ${current:,.2f}")
                print(f"  Predicted Price: ${predicted:,.2f}")
                print(f"  Prediction Difference: {price_diff_pct:+.2f}%")
                print(f"  Model R²: {r2}")
                print(f"  Model MAPE: {mape}%")
                print(f"  Model RMSE: {rmse}")
                print(f"  Validation Loss: {val_loss}")
                
                # Check confidence factors
                conf = data.get('confidence_factors', {})
                prob = data.get('probability', 0)
                print(f"  Overall Probability: {prob:.2f}")
                
                # Technical indicators analysis
                tech_indicators = conf.get('technical_indicators', [])
                print(f"  Technical Indicators Count: {len(tech_indicators)}")
                
                for indicator in tech_indicators:
                    ind_name = indicator.get('indicator', 'Unknown')
                    ind_signal = indicator.get('signal', 'N/A')
                    ind_strength = indicator.get('strength', 0)
                    print(f"    {ind_name}: {ind_signal} (strength: {ind_strength})")
                
            else:
                print(f"  Error: HTTP {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")

def check_data_freshness():
    """Check if data is fresh and recent"""
    print("\n=== Data Freshness Check ===")
    
    try:
        response = requests.get("http://localhost:8000/api/predict/BTC?timeframe=1h")
        if response.status_code == 200:
            data = response.json()
            pred_time = data.get('prediction_time', '')
            current_time = data.get('current_time', '')
            
            print(f"Prediction Time: {pred_time}")
            print(f"Current Time: {current_time}")
            
            # Parse times and check difference
            if pred_time and current_time:
                from dateutil import parser
                pred_dt = parser.parse(pred_time)
                curr_dt = parser.parse(current_time)
                time_diff = curr_dt - pred_dt
                
                print(f"Data Age: {time_diff}")
                
                if time_diff > timedelta(hours=2):
                    print("⚠️  WARNING: Data is quite old, may affect accuracy")
                else:
                    print("✅ Data is reasonably fresh")
    except Exception as e:
        print(f"Error checking data freshness: {e}")

def identify_issues():
    """Identify potential accuracy issues"""
    print("\n=== Potential Issues Identified ===")
    
    issues = []
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/api/predict/BTC?timeframe=1h")
        if response.status_code != 200:
            issues.append("API server not responding correctly")
    except:
        issues.append("Cannot connect to API server")
        
    # Check model configuration
    print("1. Model Configuration Issues:")
    print("   - 1h models have inherent difficulty due to noise")
    print("   - Negative R² indicates model struggles with volatility")
    print("   - Short lookback (48) may not capture longer trends")
    
    print("\n2. Data Quality Issues:")
    print("   - Hourly crypto data is extremely noisy")
    print("   - Technical indicators may lag in fast-moving markets")
    print("   - Limited training data (45 days) for complex patterns")
    
    print("\n3. Market Factors:")
    print("   - Crypto markets are highly unpredictable")
    print("   - External news/events not captured in technical data")
    print("   - High frequency trading creates artificial patterns")
    
    return issues

if __name__ == "__main__":
    check_prediction_accuracy()
    check_data_freshness()
    identify_issues()
