"""Timeframe configuration for the application - OPTIMIZED VERSION"""

def get_min_samples_for_timeframe(timeframe: str) -> int:
    """Calculate minimum required data points: 4 times the model's lookback period for maximum flexibility"""
    lookback = TIMEFRAME_MAP.get(timeframe, {}).get('lookback', 72)
    return lookback * 4  # Provides enough data for training, validation, and feature generation while being flexible

# This is the "Best Version" configuration for peak accuracy and speed.
TIMEFRAME_MAP = {
    "30m": {
        "period": "30d",
        "interval": "30m",
        "lookback": 48,
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "batch_size": 128,
        "learning_rate": 0.0008,
        "max_lr": 0.0015,
        "early_stopping_patience": 20,
        "technical_indicators": {
            "SMA_fast": 10,
            "SMA_slow": 30,
            "RSI": 14,
            "MACD_fast": 12,
            "MACD_slow": 26,
            "MACD_sign": 9,
            "BB_window": 20
        }
    },
    "1h": {
        "period": "60d",
        "interval": "1h",
        "lookback": 72,
        "hidden_size": 256,
        "num_layers": 4,
        "dropout": 0.3,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "max_lr": 0.001,
        "early_stopping_patience": 30,
        "technical_indicators": {
            "SMA_fast": 12,
            "SMA_slow": 24,
            "RSI": 14,
            "MACD_fast": 12,
            "MACD_slow": 26,
            "MACD_sign": 9,
            "BB_window": 20
        }
    },
    "4h": {
        "period": "120d",
        "interval": "4h",
        "lookback": 60,
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "batch_size": 64,
        "learning_rate": 0.0002,
        "max_lr": 0.0008,
        "early_stopping_patience": 40,
        "technical_indicators": {
            "SMA_fast": 7,
            "SMA_slow": 14,
            "RSI": 14,
            "MACD_fast": 12,
            "MACD_slow": 26,
            "MACD_sign": 9,
            "BB_window": 20
        }
    },
    "24h": {
        "period": "730d",
        "interval": "1d",
        "lookback": 72,
        "hidden_size": 256,
        "num_layers": 4,
        "dropout": 0.4,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "max_lr": 0.0005,
        "early_stopping_patience": 50,
        "technical_indicators": {
            "SMA_fast": 7,
            "SMA_slow": 30,
            "RSI": 14,
            "MACD_fast": 12,
            "MACD_slow": 26,
            "MACD_sign": 9,
            "BB_window": 20
        }
    }
}