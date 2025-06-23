"""Timeframe configuration for the application - OPTIMIZED VERSION"""

def get_min_samples_for_timeframe(timeframe: str) -> int:
    """Calculate minimum required data points: 4 times the model's lookback period for maximum flexibility"""
    lookback = TIMEFRAME_MAP.get(timeframe, {}).get('lookback', 72)
    return lookback * 4  # Provides enough data for training, validation, and feature generation while being flexible

TIMEFRAME_MAP = {
    "30m": {
        "period": "30d",           # INCREASED from 15d for better data quality
        "interval": "30m",
        "lookback": 48,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 128,          # OPTIMIZED for speed
        "learning_rate": 0.0001,
        "max_lr": 0.001,
        "early_stopping_patience": 20,  # OPTIMIZED for speed
        "sma_window": 5,
        "ema_span": 6,
        "bb_window": 10,
        "momentum_window": 5,
        "rsi_window": 14
    },
    "1h": {
        "period": "60d",           # INCREASED from 30d for better data quality
        "interval": "1h",
        "lookback": 72,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 64,           # OPTIMIZED for speed
        "learning_rate": 0.0001,
        "max_lr": 5e-05,
        "early_stopping_patience": 30,  # OPTIMIZED for speed
        "sma_window": 12,
        "ema_span": 9,
        "bb_window": 20,
        "momentum_window": 10,
        "rsi_window": 14
    },
    "4h": {
        "period": "120d",          # INCREASED from 90d for better data quality
        "interval": "1h",  # Fetched as 1h and resampled to 4h
        "lookback": 60,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 64,           # OPTIMIZED for speed
        "learning_rate": 2e-05,
        "max_lr": 8e-05,
        "early_stopping_patience": 40,  # OPTIMIZED for speed
        "sma_window": 5,
        "ema_span": 6,
        "bb_window": 10,
        "momentum_window": 5,
        "rsi_window": 14
    },
    "24h": {
        "period": "730d",  # Request 2 years of data for context
        "interval": "1d",
        "lookback": 72,    # CRITICAL FIX: Increased from 30 to 72 for accuracy
        "hidden_size": 192,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,           # OPTIMIZED for speed
        "learning_rate": 2e-05,
        "max_lr": 8e-05,
        "early_stopping_patience": 50,  # OPTIMIZED for speed
        "sma_window": 20,
        "ema_span": 12,
        "bb_window": 20,
        "momentum_window": 14,
        "rsi_window": 14
    }
} 