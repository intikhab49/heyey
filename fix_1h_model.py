#!/usr/bin/env python3
"""
Improved configuration for 1h BTC model to fix performance issues
"""

# Model hyperparameter suggestions for 1h timeframe
IMPROVED_1H_CONFIG = {
    "model_architecture": {
        "hidden_size": 64,  # Reduced from 128 to prevent overfitting
        "num_layers": 2,    # Reduced from 3
        "dropout": 0.3,     # Add dropout for regularization
        "bidirectional": True  # Use bidirectional LSTM
    },
    
    "training": {
        "batch_size": 32,   # Smaller batch size for better gradient updates
        "initial_lr": 1e-4, # Lower learning rate
        "max_lr": 5e-4,     # Lower max learning rate for OneCycle
        "epochs": 150,      # Fewer epochs to prevent overfitting
        "early_stopping_patience": 15,  # Earlier stopping
        "weight_decay": 1e-5,  # L2 regularization
        "gradient_clip": 1.0   # Gradient clipping
    },
    
    "data": {
        "lookback": 48,     # Reduced from 72 - less historical context
        "validation_split": 0.15,  # Smaller validation set
        "feature_selection": True,  # Enable feature selection
        "max_features": 15,  # Limit to most important features
        "normalize_method": "robust"  # Use robust scaling instead of standard
    },
    
    "ensemble": {
        "use_ensemble": True,
        "models": ["1h_model", "4h_model_downsampled"],
        "weights": [0.3, 0.7]  # Give more weight to 4h model
    }
}

# Feature importance ranking for 1h data
IMPORTANT_1H_FEATURES = [
    'Close', 'Open', 'High', 'Low', 'Volume',
    'RSI_14', 'MACD', 'MACD_Signal', 
    'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_middle',
    'EMA_20', 'SMA_20', 'ATR', 'VWAP'
]

def get_1h_model_improvements():
    """Return specific improvements for 1h model"""
    return {
        "immediate_fixes": [
            "Reduce model complexity (smaller hidden size)",
            "Add dropout regularization",
            "Use shorter lookback window",
            "Implement feature selection",
            "Lower learning rates"
        ],
        
        "alternative_approaches": [
            "Use 4h model for 1h predictions (resample)",
            "Ensemble 30m + 4h models",
            "Simple moving average + trend following",
            "Use XGBoost instead of LSTM for 1h"
        ],
        
        "data_improvements": [
            "Add more external features (market sentiment)",
            "Include order book data if available", 
            "Add volatility regime detection",
            "Use robust scaling instead of standard scaling"
        ]
    }

if __name__ == "__main__":
    improvements = get_1h_model_improvements()
    print("=== 1h Model Improvement Recommendations ===\n")
    
    for category, suggestions in improvements.items():
        print(f"{category.replace('_', ' ').title()}:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
        print()
    
    print("Current 1h Model Issues:")
    print("  • R² = -0.34 (worse than baseline)")
    print("  • Overfitting to training noise")
    print("  • Too complex for available data")
    print("  • High volatility in 1h timeframe")
    
    print("\nQuick Fix Recommendation:")
    print("  → Use the 4h model for 1h predictions")
    print("  → 4h model has R² = 0.53 (much better)")
    print("  → Interpolate 4h predictions to 1h frequency")
