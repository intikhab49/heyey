#!/usr/bin/env python3
"""
Comprehensive analysis to identify and fix model accuracy issues
Focus on data quality, feature engineering, and model architecture improvements
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import torch
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from controllers.data_fetcher import DataFetcher
from controllers.model_trainer import ModelTrainer
from ml_models.bilstm_predictor import BiLSTMWithAttention, FEATURE_LIST
from simple_config import TIMEFRAME_MAP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

async def analyze_data_quality(symbol: str, timeframe: str):
    """Analyze data quality and identify issues"""
    logger.info(f"🔍 Analyzing data quality for {symbol} {timeframe}")
    
    data_fetcher = DataFetcher()
    df = await data_fetcher.get_merged_data(symbol, timeframe)
    
    if df is None or df.empty:
        return {"error": "No data available"}
    
    analysis = {
        "data_points": len(df),
        "features": list(df.columns),
        "date_range": f"{df.index[0]} to {df.index[-1]}",
        "issues": []
    }
    
    # Check for missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    for col, pct in missing_pct.items():
        if pct > 5:
            analysis["issues"].append(f"High missing values in {col}: {pct:.1f}%")
    
    # Check for infinite values
    inf_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
            analysis["issues"].append(f"Infinite values in {col}: {inf_count}")
    
    # Check price data variance
    if 'Close' in df.columns:
        price_std = df['Close'].std()
        price_mean = df['Close'].mean()
        cv = price_std / price_mean if price_mean > 0 else float('inf')
        analysis["price_cv"] = cv
        
        if cv < 0.01:
            analysis["issues"].append(f"Very low price variance (CV={cv:.4f}) - may cause training issues")
    
    # Check feature correlations
    if len(df.columns) > 1:
        corr_matrix = df.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            analysis["issues"].append(f"High feature correlations found: {len(high_corr_pairs)} pairs")
            analysis["high_correlations"] = high_corr_pairs[:5]  # Show first 5
    
    return analysis

def test_scaler_types(data):
    """Test different scaler types to find optimal scaling"""
    scalers = {
        'RobustScaler': RobustScaler(),
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    results = {}
    
    for name, scaler in scalers.items():
        try:
            # Fit scaler
            scaled_data = scaler.fit_transform(data)
            
            # Check for issues
            has_inf = np.isinf(scaled_data).any()
            has_nan = np.isnan(scaled_data).any()
            variance = np.var(scaled_data, axis=0).mean()
            
            results[name] = {
                "has_inf": has_inf,
                "has_nan": has_nan,
                "mean_variance": variance,
                "data_range": (scaled_data.min(), scaled_data.max())
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results

async def analyze_feature_importance(symbol: str, timeframe: str):
    """Analyze feature importance and suggest improvements"""
    logger.info(f"📊 Analyzing feature importance for {symbol} {timeframe}")
    
    data_fetcher = DataFetcher()
    df = await data_fetcher.get_merged_data(symbol, timeframe)
    
    if df is None or df.empty:
        return {"error": "No data available"}
    
    # Filter to expected features
    available_features = [f for f in FEATURE_LIST if f in df.columns]
    missing_features = [f for f in FEATURE_LIST if f not in df.columns]
    
    if not available_features:
        return {"error": "No expected features found in data"}
    
    feature_df = df[available_features].copy()
    
    # Clean the data
    feature_df = feature_df.ffill().bfill()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # Calculate feature statistics
    feature_stats = {}
    target = feature_df['Close'] if 'Close' in feature_df.columns else feature_df.iloc[:, 0]
    
    for feature in available_features:
        if feature == 'Close':
            continue
        
        try:
            # Calculate correlation with target
            corr = feature_df[feature].corr(target)
            
            # Calculate feature variance
            variance = feature_df[feature].var()
            
            # Calculate feature range
            feature_range = feature_df[feature].max() - feature_df[feature].min()
            
            feature_stats[feature] = {
                "correlation": corr if not np.isnan(corr) else 0,
                "variance": variance if not np.isnan(variance) else 0,
                "range": feature_range if not np.isnan(feature_range) else 0,
                "null_count": feature_df[feature].isnull().sum()
            }
        except Exception as e:
            feature_stats[feature] = {"error": str(e)}
    
    # Rank features by absolute correlation
    ranked_features = sorted(
        [(k, v) for k, v in feature_stats.items() if isinstance(v, dict) and 'correlation' in v],
        key=lambda x: abs(x[1]['correlation']),
        reverse=True
    )
    
    return {
        "total_features": len(FEATURE_LIST),
        "available_features": len(available_features),
        "missing_features": missing_features,
        "feature_stats": dict(ranked_features[:10]),  # Top 10 features
        "low_variance_features": [k for k, v in feature_stats.items() 
                                 if isinstance(v, dict) and v.get('variance', 1) < 0.001]
    }

async def suggest_improvements(symbol: str, timeframe: str):
    """Suggest specific improvements based on analysis"""
    logger.info(f"💡 Generating improvement suggestions for {symbol} {timeframe}")
    
    # Run all analyses
    data_analysis = await analyze_data_quality(symbol, timeframe)
    feature_analysis = await analyze_feature_importance(symbol, timeframe)
    
    suggestions = []
    
    # Data quality suggestions
    if data_analysis.get("issues"):
        suggestions.append({
            "category": "Data Quality",
            "priority": "HIGH",
            "issues": data_analysis["issues"],
            "recommendations": [
                "Implement better data cleaning and outlier detection",
                "Add more robust missing value handling",
                "Consider using different data sources for missing periods"
            ]
        })
    
    # Feature engineering suggestions
    if feature_analysis.get("missing_features"):
        suggestions.append({
            "category": "Feature Engineering",
            "priority": "MEDIUM",
            "missing_features": feature_analysis["missing_features"],
            "recommendations": [
                "Implement missing technical indicators",
                "Add feature importance-based selection",
                "Consider domain-specific crypto features (funding rates, social sentiment)"
            ]
        })
    
    # Model architecture suggestions
    config = TIMEFRAME_MAP.get(timeframe, {})
    suggestions.append({
        "category": "Model Architecture",
        "priority": "MEDIUM",
        "current_config": config,
        "recommendations": [
            "Try different sequence lengths for different timeframes",
            "Experiment with transformer-based architectures",
            "Add regularization techniques (L1/L2, early stopping)",
            "Consider ensemble methods"
        ]
    })
    
    # Scaling suggestions
    data_fetcher = DataFetcher()
    df = await data_fetcher.get_merged_data(symbol, timeframe)
    if df is not None and not df.empty:
        available_features = [f for f in FEATURE_LIST if f in df.columns]
        if available_features:
            feature_df = df[available_features].ffill().bfill()
            scaler_results = test_scaler_types(feature_df.values)
            
            best_scaler = min(scaler_results.items(), 
                            key=lambda x: x[1].get('has_inf', True) + x[1].get('has_nan', True))
            
            suggestions.append({
                "category": "Data Scaling",
                "priority": "HIGH",
                "scaler_test_results": scaler_results,
                "recommended_scaler": best_scaler[0],
                "recommendations": [
                    f"Use {best_scaler[0]} for better numerical stability",
                    "Add gradient clipping to prevent exploding gradients",
                    "Normalize features to similar scales"
                ]
            })
    
    return suggestions

async def main():
    """Run comprehensive model analysis"""
    test_cases = [
        ("BTC", "24h"),
        ("ETH", "1h"),
        ("DOGE", "4h")
    ]
    
    print("🚀 Comprehensive Model Analysis Started")
    print("=" * 60)
    
    for symbol, timeframe in test_cases:
        print(f"\n📈 Analyzing {symbol} {timeframe}")
        print("-" * 40)
        
        try:
            # Data quality analysis
            data_analysis = await analyze_data_quality(symbol, timeframe)
            print(f"Data points: {data_analysis.get('data_points', 'N/A')}")
            print(f"Features: {len(data_analysis.get('features', []))}")
            if data_analysis.get('issues'):
                print(f"⚠️  Issues found: {len(data_analysis['issues'])}")
                for issue in data_analysis['issues'][:3]:
                    print(f"   - {issue}")
            else:
                print("✅ No major data quality issues")
            
            # Feature analysis
            feature_analysis = await analyze_feature_importance(symbol, timeframe)
            print(f"Available features: {feature_analysis.get('available_features', 'N/A')}")
            if feature_analysis.get('missing_features'):
                print(f"⚠️  Missing features: {len(feature_analysis['missing_features'])}")
            
            # Improvement suggestions
            suggestions = await suggest_improvements(symbol, timeframe)
            print(f"\n💡 Improvement suggestions: {len(suggestions)}")
            for suggestion in suggestions:
                print(f"   {suggestion['category']} ({suggestion['priority']})")
        
        except Exception as e:
            print(f"❌ Error analyzing {symbol} {timeframe}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 Analysis Complete")

if __name__ == "__main__":
    asyncio.run(main())
