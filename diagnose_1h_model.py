#!/usr/bin/env python3
"""
Diagnose issues with the 1h BTC model performance
"""

import sys
import os
sys.path.append('/home/intikhab/CryptoAion-AI-main/Crypto-main/CryptoAion_AI_project_runnable')

import pandas as pd
import numpy as np
from controllers.data_fetcher import DataFetcher

def analyze_1h_model():
    """Analyze the 1h model data and training quality"""
    print("=== BTC 1h Model Analysis ===\n")
    
    # Initialize components
    data_fetcher = DataFetcher()
    
    try:
        # 1. Fetch 1h data
        print("1. Fetching 1h data...")
        data = data_fetcher.fetch_data('BTC', '1h')
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {list(data.columns)}")
        
        # 2. Check price statistics
        print("\n2. Price Statistics:")
        close_prices = data['Close']
        print(f"Mean price: ${close_prices.mean():,.2f}")
        print(f"Std price: ${close_prices.std():,.2f}")
        print(f"Min price: ${close_prices.min():,.2f}")
        print(f"Max price: ${close_prices.max():,.2f}")
        print(f"Price volatility (CV): {close_prices.std()/close_prices.mean():.4f}")
        
        # 3. Check for data quality issues
        print("\n3. Data Quality Check:")
        print(f"Missing values: {data.isnull().sum().sum()}")
        print(f"Duplicate timestamps: {data.index.duplicated().sum()}")
        
        # Price changes analysis
        price_changes = close_prices.pct_change().dropna()
        print(f"Price change mean: {price_changes.mean():.6f}")
        print(f"Price change std: {price_changes.std():.6f}")
        print(f"Large moves (>5%): {(abs(price_changes) > 0.05).sum()}")
        
        # 4. Technical indicators quality
        print("\n4. Technical Indicators Analysis:")
        if 'RSI_14' in data.columns:
            rsi = data['RSI_14'].dropna()
            print(f"RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
            print(f"RSI current: {rsi.iloc[-1]:.2f}")
        
        if 'MACD' in data.columns:
            macd = data['MACD'].dropna()
            print(f"MACD range: {macd.min():.2f} - {macd.max():.2f}")
            print(f"MACD current: {macd.iloc[-1]:.2f}")
        
        # 5. Check recent price trend
        print("\n5. Recent Price Trend (last 24 hours):")
        recent_data = data.tail(24)  # Last 24 hours
        recent_prices = recent_data['Close']
        trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
        print(f"24h trend: {trend:.2f}%")
        
        # 6. Volatility analysis by hour
        print("\n6. Hourly Volatility Pattern:")
        data['Hour'] = data.index.hour
        hourly_vol = data.groupby('Hour')['Close'].apply(lambda x: x.pct_change().std()).dropna()
        print("Highest volatility hours:")
        print(hourly_vol.sort_values(ascending=False).head(5))
        
        # 7. Model performance prediction
        print("\n7. Potential Model Issues:")
        
        # Check if there are enough data points
        if len(data) < 500:
            print("⚠️  Insufficient data points for robust training")
        
        # Check volatility
        if price_changes.std() > 0.05:
            print("⚠️  High volatility may cause training instability")
        
        # Check for trend changes
        recent_trend = price_changes.tail(72).mean()  # Last 3 days
        if abs(recent_trend) < 0.001:
            print("⚠️  Very low trend signal - model may struggle to learn patterns")
        
        # Check indicator correlations
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 5:
            corr_matrix = data[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                print("⚠️  Highly correlated features detected (may cause overfitting):")
                for feat1, feat2, corr in high_corr_pairs[:3]:
                    print(f"   {feat1} <-> {feat2}: {corr:.3f}")
        
        print("\n8. Recommendations:")
        print("- Consider using 4h model (better performance) for 1h predictions")
        print("- Increase lookback window for more context")
        print("- Add ensemble prediction combining multiple timeframes")
        print("- Implement feature selection to reduce overfitting")
        print("- Use regularization techniques (dropout, weight decay)")
        
        return data
        
    except Exception as e:
        print(f"Error analyzing 1h model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_1h_model()
