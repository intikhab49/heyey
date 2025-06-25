"""Complete configuration for CryptoAion"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Get API key from environment, strip any comments
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '').split('#')[0].strip()
COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'

# Core settings
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(os.path.dirname(__file__), "models"))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', '0.2'))
CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes
DEFAULT_TIMEFRAME = "24h"
TIMEFRAME_OPTIONS = ["1m", "5m", "15m", "30m", "1h", "4h", "24h"]

# Timeframe configuration map - EXPERT TUNED FOR SPEED & ACCURACY
# This is the "Best Version" configuration for peak accuracy and speed.
TIMEFRAME_MAP = {
    "30m": {
        "period": "30d",                   # Fetch 30 days of data
        "lookback": 48,                    # Look at the last 24 hours
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 128,                 # INCREASED for speed
        "learning_rate": 0.0001,
        "max_lr": 0.001,
        "early_stopping_patience": 20      # REDUCED for speed
    },
    "1h": {
        "period": "60d",                   # Fetch 60 days of data
        "lookback": 72,                    # Look at the last 3 days
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 64,                  # INCREASED for speed
        "learning_rate": 0.0001,
        "max_lr": 5e-05,
        "early_stopping_patience": 30      # Tuned for 1h
    },
    "4h": {
        "period": "120d",                  # Fetch 120 days of data
        "lookback": 60,                    # Look at the last 10 days
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 64,                  # INCREASED for speed
        "learning_rate": 2e-05,
        "max_lr": 8e-05,
        "early_stopping_patience": 40
    },
    "24h": {
        "period": "730d",                  # Fetch 2 years of data
        "lookback": 72,                    # CRITICAL FIX: Look at last ~2.5 months for accuracy
        "hidden_size": 192,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,                  # INCREASED for speed
        "learning_rate": 2e-05,
        "max_lr": 8e-05,
        "early_stopping_patience": 50
    }
}

# API and rate limiting settings
MAX_REQUESTS_PER_MINUTE = 60
# Alpha Vantage disabled for crypto-focused system
# ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')

# Create necessary directories
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

# Feature list for compatibility with old models/tests
FEATURE_LIST = [
    'open', 'high', 'low', 'close', 'volume',
    'sma', 'ema', 'bb_upper', 'bb_middle', 'bb_lower',
    'rsi', 'macd', 'signal', 'histogram', 'momentum',
    'price_change', 'volume_change', 'volatility',
    'price_range', 'bb_position', 'rsi_normalized',
    'macd_normalized', 'volume_ratio', 'price_momentum',
    'trend_strength', 'support_resistance'
]

# Settings class for compatibility 
class SimpleSettings:
    """Settings object for compatibility with old config usage"""
    def __init__(self):
        self.MODEL_PATH = MODEL_PATH
        self.BATCH_SIZE = BATCH_SIZE
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.CACHE_TTL = CACHE_TTL
        self.DEFAULT_TIMEFRAME = DEFAULT_TIMEFRAME
        self.TIMEFRAME_OPTIONS = TIMEFRAME_OPTIONS
        self.MAX_REQUESTS_PER_MINUTE = MAX_REQUESTS_PER_MINUTE
        self.COINGECKO_API_KEY = COINGECKO_API_KEY
        self.COINGECKO_API_URL = COINGECKO_API_URL
        
        # Additional settings that might be referenced
        self.MIN_PRICE_POINTS = 100
        self.MAX_RETRIES = 3
        self.REQUEST_TIMEOUT = 30
        self.ENABLE_CACHING = True
        
        # WebSocket settings
        self.WS_UPDATE_INTERVAL = {"30m": 30, "1h": 60, "4h": 240, "24h": 1440}
        self.WS_MAX_ERRORS = 5
        
        # Database settings (if needed for compatibility)
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./crypto_data.db')
        
        # JWT Authentication settings
        self.JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-this-in-production')
        self.JWT_ALGORITHM = 'HS256'
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create settings instance for compatibility
settings = SimpleSettings()

# TORTOISE_ORM config for compatibility with migrations
TORTOISE_ORM = {
    "connections": {"default": settings.DATABASE_URL},
    "apps": {
        "models": {
            "models": ["models", "aerich.models"],
            "default_connection": "default",
        },
    },
}

print(f"Loaded API key: {COINGECKO_API_KEY[:10]}..." if COINGECKO_API_KEY else "No API key loaded")
