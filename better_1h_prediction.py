#!/usr/bin/env python3
"""
Simple fallback for 1h predictions using the good 4h model
"""

import sys
import os
sys.path.append('/home/intikhab/CryptoAion-AI-main/Crypto-main/CryptoAion_AI_project_runnable')

import requests
import json

def get_better_1h_prediction(symbol="BTC"):
    """Get a 1h prediction using the superior 4h model"""
    try:
        # Get 4h prediction
        response = requests.get(f"http://localhost:8000/api/predict/{symbol}?timeframe=4h&force_retrain=false")
        if response.status_code == 200:
            result = response.json()
            
            # Modify the response to indicate it's for 1h but using 4h model
            result["original_timeframe"] = "4h"
            result["requested_timeframe"] = "1h"
            result["timeframe"] = "1h"
            result["note"] = "Using superior 4h model (R² = 0.53) instead of poor 1h model (R² = -0.94)"
            result["model_source"] = "4h_model_for_1h_prediction"
            
            print("=== Better 1h Prediction (Using 4h Model) ===")
            print(f"Symbol: {result['symbol']}")
            print(f"Predicted Price: ${result['predicted_price']:,.2f}")
            print(f"Last Price: ${result['last_price']:,.2f}")
            print(f"Change: ${result['predicted_price'] - result['last_price']:,.2f}")
            print(f"Change %: {((result['predicted_price'] / result['last_price']) - 1) * 100:.2f}%")
            print(f"Probability: {result['probability']:.1%}")
            print(f"Note: {result['note']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return None

if __name__ == "__main__":
    get_better_1h_prediction()
