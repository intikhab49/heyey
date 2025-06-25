"""
This module handles the core logic for cryptocurrency price prediction, including:
- Fetching and preparing data for training.
- Defining, training, and evaluating the predictive model (LSTM).
- Managing model lifecycle, including saving, loading, and versioning.
- Providing prediction endpoints.
"""

# Standard Library Imports
import logging
import sys
import time
from datetime import datetime
from typing import Optional, Dict

# Third-party Imports
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
import torch
import torch.nn as nn
from cachetools import TTLCache
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from torch.utils.data import DataLoader, TensorDataset

# Optional hyperparameter tuning support
try:
    import optuna  # type: ignore
except ImportError:
    optuna = None

import asyncio  # Used by hyperparameter tuner

# Project-specific Imports
import simple_config
from controllers.data_fetcher import DataFetcher
from controllers.data_validator import DataValidator, rate_limit
from controllers.error_handler import ErrorCategory, ErrorSeverity, ErrorTracker
from controllers.feature_engineering import calculate_technical_indicators
from controllers.model_manager import ModelManager
from ml_models.bilstm_predictor import BiLSTMWithAttention
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def get_next_timestamp(last_timestamp, timeframe):
    """Calculate next timestamp based on timeframe with timezone handling"""
    timeframe_deltas = {
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "24h": pd.Timedelta(days=1),
    }
    now = datetime.now(pytz.UTC)
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize("UTC")
    if timeframe == "24h":
        next_ts = (last_timestamp + pd.Timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        next_ts = last_timestamp + timeframe_deltas[timeframe]
    if next_ts <= now:
        periods = int((now - last_timestamp) / timeframe_deltas[timeframe]) + 1
        if timeframe == "24h":
            next_ts = (last_timestamp + periods * timeframe_deltas[timeframe]).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            next_ts = last_timestamp + periods * timeframe_deltas[timeframe]
    return next_ts


# Allowlist datetime.datetime for safe model loading
torch.serialization.add_safe_globals([datetime])

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants from simple_config - Optimized for Accuracy
MODEL_PATH = simple_config.MODEL_PATH
MAX_RETRIES = 3
RETRY_DELAY = 2
EPOCHS = 300  # Increased from 200 for better accuracy
PATIENCE = 40  # Increased patience for better convergence
CACHE_DIR = "cache"
BATCH_SIZE = simple_config.BATCH_SIZE
MIN_MEMORY_GB = 0.75
VALIDATION_SPLIT = simple_config.VALIDATION_SPLIT

# Cache for sentiment and price data
SENTIMENT_CACHE = TTLCache(maxsize=100, ttl=simple_config.CACHE_TTL)
PRICE_CACHE = TTLCache(maxsize=10, ttl=60)

# CoinGecko ID mapping
# Global cache for CoinGecko IDs to avoid repeated API calls
coin_id_cache = {}


async def get_coingecko_id(symbol: str) -> Optional[str]:
    """Dynamically get CoinGecko ID from symbol using predefined mapping for major cryptocurrencies"""
    try:
        symbol = symbol.upper()

        # Predefined mapping for major cryptocurrencies to avoid confusion
        major_crypto_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDT": "tether",
            "BNB": "binancecoin",
            "SOL": "solana",
            "XRP": "ripple",
            "USDC": "usd-coin",
            "STETH": "staked-ether",
            "ADA": "cardano",
            "AVAX": "avalanche-2",
            "DOGE": "dogecoin",
            "TRX": "tron",
            "TON": "the-open-network",
            "SHIB": "shiba-inu",
            "DOT": "polkadot",
            "WBTC": "wrapped-bitcoin",
            "BCH": "bitcoin-cash",
            "LINK": "chainlink",
            "LTC": "litecoin",
            "MATIC": "matic-network",
            "UNI": "uniswap",
            "LEO": "leo-token",
            "XLM": "stellar",
            "ETC": "ethereum-classic",
            "OKB": "okb",
            "ICP": "internet-computer",
            "ATOM": "cosmos",
            "FIL": "filecoin",
            "HBAR": "hedera-hashgraph",
            "CRO": "crypto-com-chain",
            "ARB": "arbitrum",
            "VET": "vechain",
            "MNT": "mantle",
            "DAI": "dai",
            "IMX": "immutable-x",
            "APT": "aptos",
            "OP": "optimism",
            "NEAR": "near",
            "TUSD": "true-usd",
            "INJ": "injective-protocol",
            "FIRST": "first-digital-usd",
            "GRT": "the-graph",
            "LDO": "lido-dao",
            "MKR": "maker",
            "AAVE": "aave",
            "SNX": "synthetix-network-token",
            "QNT": "quant-network",
            "RUNE": "thorchain",
            "FLOW": "flow",
            "SAND": "the-sandbox",
            "MANA": "decentraland",
            "APE": "apecoin",
            "ALGO": "algorand",
            "EGLD": "elrond-erd-2",
            "XTZ": "tezos",
            "CHZ": "chiliz",
            "EOS": "eos",
            "AXS": "axie-infinity",
            "THETA": "theta-token",
            "FTM": "fantom",
            "ICP": "internet-computer",
            "XMR": "monero",
            "KLAY": "klay-token",
            "BSV": "bitcoin-sv",
            "NEO": "neo",
            "USDD": "usdd",
            "FRAX": "frax",
            "CAKE": "pancakeswap-token",
            "CRV": "curve-dao-token",
            "ZEC": "zcash",
            "DASH": "dash",
        }

        # Check predefined mapping first
        if symbol in major_crypto_mapping:
            coin_id = major_crypto_mapping[symbol]
            logger.info(
                "Found CoinGecko ID for %s from predefined mapping: %s", symbol, coin_id
            )
            return coin_id

        # Check cache for non-major cryptocurrencies
        if symbol in coin_id_cache:
            logger.debug(
                "Found cached CoinGecko ID for %s: %s", symbol, coin_id_cache[symbol]
            )
            return coin_id_cache[symbol]

        # For non-major cryptocurrencies, use API call
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            logger.warning(
                "Failed to fetch CoinGecko coins list: %s", response.status_code
            )
            return None

        coins_data = response.json()

        # Search for the symbol in the coins list (only for non-major coins)
        for coin in coins_data:
            if coin.get("symbol", "").upper() == symbol:
                coin_id = coin.get("id")
                coin_id_cache[symbol] = coin_id  # Cache the result
                logger.info("Found CoinGecko ID for %s: %s", symbol, coin_id)
                return coin_id

        logger.warning("No CoinGecko ID found for symbol: %s", symbol)
        return None

    except Exception as e:
        logger.error("Error getting CoinGecko ID for %s: %s", symbol, e)
        return None


# Add after imports
model_manager = ModelManager()
error_tracker = ErrorTracker()


# New synchronous helper function for training data - CHANGING TO ASYNC
async def _get_and_prepare_training_data(
    symbol: str,
    timeframe: str,
    lookback: int,
    forecast_horizon: int = 1,
    test_size: float = simple_config.VALIDATION_SPLIT,
    scaler_type: str = "standard",
    reference_features: Optional[np.ndarray] = None,
) -> tuple:

    """Get and prepare training data with proper tensor shapes"""
    try:
        logger.info(
            "Starting training data preparation for %s (%s), lookback=%s",
            symbol,
            timeframe,
            lookback,
        )

        # Get historical data and prepare features
        features_df, target_series = await prepare_features(symbol, timeframe)

        # Get the definitive list of features for model input
        feature_column_names_for_model_input = list(features_df.columns)

        logger.info(
            "Using definitive feature columns for model input: %s",
            feature_column_names_for_model_input,
        )

        # Ensure all columns designated for model input are actually in features_df
        missing_cols_in_df = [
            col
            for col in feature_column_names_for_model_input
            if col not in features_df.columns
        ]
        if missing_cols_in_df:
            raise ValueError(f"Missing columns in features_df: {missing_cols_in_df}")

        # Extract features and target
        X_data_to_sequence = features_df[feature_column_names_for_model_input].values
        y_data = target_series.values

        # Log data shapes before scaling
        logger.info(
            "[_GET_AND_PREPARE_TRAINING_DATA] Shape of data PASSED to scaler.fit() (X_data_to_sequence): %s",
            X_data_to_sequence.shape,
        )
        logger.info(
            "[_GET_AND_PREPARE_TRAINING_DATA] Columns of data PASSED to scaler.fit() (X_data_to_sequence): %s",
            feature_column_names_for_model_input,
        )

        # Select feature scaler based on user choice
        scaler_type = scaler_type.lower()
        if scaler_type == "robust":
            feature_scaler = RobustScaler()
            target_scaler = RobustScaler()
        elif scaler_type == "minmax":
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        elif scaler_type == "none":
            feature_scaler = None
            target_scaler = None
            X_data_scaled = X_data_to_sequence.copy()
            y_data_scaled = y_data.reshape(-1, 1)
        else:
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()

        if feature_scaler is not None:
            X_data_scaled = feature_scaler.fit_transform(X_data_to_sequence)
            y_data_scaled = target_scaler.fit_transform(y_data.reshape(-1, 1))
        else:
            # No scaling requested
            X_data_scaled = X_data_to_sequence
            y_data_scaled = y_data.reshape(-1, 1)
        y_data_scaled = target_scaler.fit_transform(y_data.reshape(-1, 1))

        # Basic target distribution diagnostics
        pos_ratio = float(np.mean(y_data > 0))
        if pos_ratio < 0.2 or pos_ratio > 0.8:
            logger.warning(
                "[IMBALANCE_CHECK] Target directional imbalance detected (positive ratio = %.2f)",
                pos_ratio,
            )

        if feature_scaler is not None:
            logger.info(
                "[_GET_AND_PREPARE_TRAINING_DATA] Using %s scaler for features/target",
                scaler_type,
            )
            logger.info(
                "[_GET_AND_PREPARE_TRAINING_DATA] Target scaling - center: %s, scale: %s",
                target_scaler.mean_[0],
                target_scaler.scale_[0],
            )
            logger.info(
                "[_GET_AND_PREPARE_TRAINING_DATA] Target range before scaling - min: %s, max: %s",
                y_data.min(),
                y_data.max(),
            )
            logger.info(
                "[_GET_AND_PREPARE_TRAINING_DATA] Target range after scaling - min: %s, max: %s",
                y_data_scaled.min(),
                y_data_scaled.max(),
            )

        # Create sequences
        X_sequences = []
        y_sequences = []

        # Create lookback/forecast sequences
        max_idx = len(X_data_scaled) - lookback - forecast_horizon + 1
        for i in range(max_idx):
            X_sequences.append(X_data_scaled[i : (i + lookback)])
            y_sequences.append(y_data_scaled[i + lookback + forecast_horizon - 1])

        # Convert to numpy arrays with correct shapes
        X_sequences = np.array(X_sequences)  # Shape: (n_samples, lookback, n_features)
        y_sequences = np.array(y_sequences)  # Shape: (n_samples, 1)

        # Optional data drift detection against reference set
        if reference_features is not None:
            try:
                drift = check_data_drift(reference_features, X_data_scaled, feature_column_names_for_model_input)
                logger.info("[DATA_DRIFT] Drift detected: %s", drift)
            except Exception as drift_err:
                logger.warning("[DATA_DRIFT] Detection failed: %s", drift_err)

        # Split into training and validation sets
        split_idx = int(len(X_sequences) * (1 - test_size))

        X_train = X_sequences[:split_idx]
        y_train = y_sequences[:split_idx]
        X_val = X_sequences[split_idx:]
        y_val = y_sequences[split_idx:]

        # Convert to PyTorch tensors with correct shapes
        X_train = torch.FloatTensor(X_train)  # Shape: (n_train, lookback, n_features)
        y_train = torch.FloatTensor(y_train).view(-1, 1)  # Shape: (n_train, 1)
        X_val = torch.FloatTensor(X_val)  # Shape: (n_val, lookback, n_features)
        y_val = torch.FloatTensor(y_val).view(-1, 1)  # Shape: (n_val, 1)

        # Log final shapes
        logger.info("Training data: X_train %s, y_train %s", X_train.shape, y_train.shape)
        logger.info("Validation data: X_val %s, y_val %s", X_val.shape, y_val.shape)

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            feature_scaler,
            target_scaler,
            feature_column_names_for_model_input,
            features_df,
        )

    except Exception as e:
        logger.error("Error in _get_and_prepare_training_data: %s", str(e))
        raise


@rate_limit(api_name="yfinance")
def fetch_yfinance_data(
    symbol, period, interval, timeframe: str = None
):  # ADDED timeframe ARG
    """Cached yfinance data fetch with improved error handling and data validation"""
    try:
        # This part still assumes direct yfinance call or that DataFetcher aligns
        ticker_obj = yf.Ticker(f"{symbol.upper()}-USD")
        data = ticker_obj.history(period=period, interval=interval, auto_adjust=True)

        if data is None or data.empty:
            error_id = error_tracker.track_error(
                ValueError(f"Could not fetch yfinance data for {symbol}"),
                ErrorSeverity.ERROR,
                ErrorCategory.DATA_FETCH,
                "yfinance_direct_fetcher",  # Changed source
                metadata={"symbol": symbol, "period": period, "interval": interval},
            )
            # raise ValueError(f"Data fetch failed (Error ID: {error_id})") # Allow fallback
            logger.error(
                "YFinance direct fetch for %s returned no data (Error ID: %s).", symbol, error_id
            )
            return pd.DataFrame()

        # Standardize column names
        data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
            errors="ignore",
        )
        data.columns = data.columns.str.capitalize()

        # Determine the correct timeframe key for DataValidator
        validator_timeframe_key = timeframe  # Use the passed 'timeframe' (e.g., "24h")
        if (
            not validator_timeframe_key
            or validator_timeframe_key not in simple_config.TIMEFRAME_MAP
        ):
            # Fallback or mapping if the main 'timeframe' isn't directly usable
            # This logic might need to be more robust if 'interval' can be varied independently
            logger.warning(
                "Validator timeframe key '%s' is invalid or not in TIMEFRAME_MAP. Attempting to use 'interval' ('%s') or a default.",
                validator_timeframe_key if validator_timeframe_key else 'None',
                interval
            )
            # Simple fallback: if interval is a key, use it. Otherwise, try to map or use default.
            if interval in simple_config.TIMEFRAME_MAP:
                validator_timeframe_key = interval
            else:  # Last resort, try to find a match or use default. This could be improved.
                compatible_key = next(
                    (
                        k
                        for k, v in simple_config.TIMEFRAME_MAP.items()
                        if v.get("interval") == interval
                    ),
                    None,
                )
                if compatible_key:
                    validator_timeframe_key = compatible_key
                else:
                    validator_timeframe_key = simple_config.DEFAULT_TIMEFRAME
                    logger.error(
                        "Could not map yfinance interval '%s' to a valid TIMEFRAME_MAP key. Using default: '%s'",
                        interval, validator_timeframe_key
                    )

        logger.debug(
            "DataValidator will use timeframe_key: '%s' (derived from input timeframe: '%s', interval: '%s')",
            validator_timeframe_key, timeframe, interval
        )
        validator = DataValidator(timeframe=validator_timeframe_key)
        data, metrics = validator.validate_and_clean_data(data)

        if metrics.overall_quality < 0.7:
            error_id = error_tracker.track_error(
                ValueError(
                    f"Data quality below threshold. Metrics: {metrics.to_dict()}"
                ),
                ErrorSeverity.WARNING,
                ErrorCategory.DATA_QUALITY,
                "data_validator_yfinance",
                metadata={"quality_metrics": metrics.to_dict()},
            )
            logger.warning(
                "Low quality data (Error ID: %s). Metrics: %s", error_id, metrics.to_dict()
            )

        return data

    except Exception as e:
        error_id = error_tracker.track_error(
            e,
            ErrorSeverity.ERROR,
            ErrorCategory.DATA_FETCH,
            "fetch_yfinance_data_general_error",
            metadata={
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "error_details": str(e),
            },
        )
        logger.error("Error fetching yfinance data (Error ID: %s): %s", error_id, str(e))
        # raise # Or return empty pd.DataFrame() to allow fallback
        return pd.DataFrame()


def check_server(timeout: int = 2) -> bool:
    """Check if FastAPI server is running"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        logger.error("FastAPI server is not running")
        return False


@rate_limit(api_name="coingecko")
async def fetch_coingecko_price(symbol: str):
    """Fetch real-time price from CoinGecko with improved error handling"""
    cache_key = f"price_{symbol}"
    if cache_key in PRICE_CACHE:
        logger.debug("Returning cached price for %s", symbol)
        return PRICE_CACHE[cache_key]

    coin_id = await get_coingecko_id(symbol)
    if not coin_id:
        error_id = error_tracker.track_error(
            ValueError(f"No CoinGecko ID mapping for {symbol}"),
            ErrorSeverity.ERROR,
            ErrorCategory.DATA_FETCH,
            "coingecko_fetcher",
        )
        logger.error("Invalid symbol (Error ID: %s)", error_id)
        return None

    url = f"{simple_config.COINGECKO_API_URL}/simple/price"
    headers = {"accept": "application/json"}
    api_key = simple_config.COINGECKO_API_KEY
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

    params = {"ids": coin_id, "vs_currencies": "usd", "include_24hr_change": "true"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if coin_id in data and "usd" in data[coin_id]:
                price = float(data[coin_id]["usd"])
                PRICE_CACHE[cache_key] = price
                logger.debug("Fetched price from API for %s: %s", symbol, price)
                return price
        else:
            error_message = f"API request failed with status {response.status_code}. Response: {response.text}"
            error_id = error_tracker.track_error(
                Exception(error_message),
                ErrorSeverity.ERROR,
                ErrorCategory.API,
                "coingecko_api",
            )
            logger.error("API request failed (Error ID: %s)", error_id)

    except requests.exceptions.RequestException as e:
        error_id = error_tracker.track_error(
            e, ErrorSeverity.ERROR, ErrorCategory.API, "coingecko_api"
        )
        logger.error("API request error (Error ID: %s): %s", error_id, str(e))

    return None


async def fetch_coingecko_sentiment(symbol: str):
    """Fetch sentiment from CoinGecko with improved error handling"""
    cache_key = f"sentiment_{symbol}"
    if cache_key in SENTIMENT_CACHE:
        logger.debug("Returning cached sentiment for %s", symbol)
        return SENTIMENT_CACHE[cache_key]

    coin_id = await get_coingecko_id(symbol)
    if not coin_id:
        logger.warning(
            "No CoinGecko mapping for symbol %s, using default sentiment", symbol
        )
        return 0.5

    url = f"{simple_config.COINGECKO_API_URL}/coins/{coin_id}"
    headers = {"accept": "application/json"}
    api_key = simple_config.COINGECKO_API_KEY
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

    try:
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            sentiment = data.get("sentiment_votes_up_percentage", 50) / 100
            logger.debug("Fetched sentiment from API for %s: %s", symbol, sentiment)
            SENTIMENT_CACHE[cache_key] = sentiment
            return sentiment
        else:
            error_message = f"API request failed with status {response.status_code}. Response: {response.text}"
            logger.error("API request failed: %s", error_message)  # Simplified logging for now, can re-add ErrorTracker if needed

    except requests.exceptions.RequestException as e:
        logger.error("API request error: %s", str(e))  # Simplified logging for now

    return 0.5  # Default neutral sentiment


@rate_limit(api_name="coingecko")
async def fetch_coingecko_fallback(coin_id, timeframe):
    """Fallback to CoinGecko historical data with improved rate limiting"""
    try:
        # Use period from TIMEFRAME_MAP
        days_str = simple_config.TIMEFRAME_MAP[timeframe].get("period", "30d")
        days = int(days_str[:-1])  # Convert '30d' to 30

        # Use the dynamic lookup instead of hardcoded mapping
        mapped_coin_id = await get_coingecko_id(coin_id)
        if not mapped_coin_id:
            mapped_coin_id = coin_id.lower()  # fallback to lowercase

        url = f"{simple_config.COINGECKO_API_URL}/coins/{mapped_coin_id}/market_chart"
        headers = {"accept": "application/json"}
        api_key = simple_config.COINGECKO_API_KEY
        if api_key:
            headers["x-cg-demo-api-key"] = api_key

        params = {
            "vs_currency": "usd",
            "days": str(days),
            "interval": "hourly" if days <= 90 else "daily",
        }

        backoff = 1  # Initial backoff time in seconds
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=15)

                if resp.status_code == 200:
                    data = resp.json()
                    if not data or "prices" not in data:
                        logger.error(
                            "Invalid response format from CoinGecko for %s", coin_id
                        )
                        return pd.DataFrame()

                    # Convert data to DataFrame
                    prices = data["prices"]
                    volumes = data.get("total_volumes", [[0, 0]] * len(prices))

                    df = pd.DataFrame(prices, columns=["timestamp", "price"])
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit="ms", utc=True
                    )
                    df.set_index("timestamp", inplace=True)

                    # Add volume data
                    volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
                    volume_df["timestamp"] = pd.to_datetime(
                        volume_df["timestamp"], unit="ms", utc=True
                    )
                    volume_df.set_index("timestamp", inplace=True)

                    # Create OHLCV DataFrame
                    final_df = pd.DataFrame(index=df.index)
                    final_df["Close"] = df["price"]
                    final_df["Volume"] = volume_df["volume"]
                    final_df["Open"] = final_df["Close"].shift(1)
                    final_df["High"] = final_df["Close"].rolling(2, min_periods=1).max()
                    final_df["Low"] = final_df["Close"].rolling(2, min_periods=1).min()

                    # Forward fill any missing data
                    final_df = final_df.ffill().bfill()

                    logger.debug(
                        "CoinGecko fallback fetched %d rows for %s", len(final_df), coin_id
                    )
                    return final_df

                elif resp.status_code == 429:  # Rate limit hit
                    wait_time = backoff * (2**attempt)
                    logger.warning("Rate limit hit, waiting %ds before retry", wait_time)
                    time.sleep(wait_time)
                    continue

                else:
                    logger.error(
                        "CoinGecko fallback failed with status code %d", resp.status_code
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return pd.DataFrame()

            except requests.exceptions.RequestException as e:
                logger.error("Request error on attempt %d: %s", attempt + 1, str(e))
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue

        logger.warning(
            "CoinGecko fallback failed for %s after %d attempts", coin_id, MAX_RETRIES
        )
        return pd.DataFrame()

    except Exception as e:
        logger.error("CoinGecko fallback failed: %s", str(e))
        return pd.DataFrame()


async def get_latest_data(symbol: str, timeframe: str = "24h") -> pd.DataFrame:  # ASYNC
    try:
        data_fetcher = DataFetcher()
        data = await data_fetcher.get_merged_data(symbol, timeframe)  # AWAITED

        if data is None or data.empty:
            logger.error(
                "Data fetch for %s timeframe %s returned None or empty after get_merged_data.",
                symbol,
                timeframe,
            )
            raise ValueError(
                f"Could not fetch sufficient data for {symbol} (timeframe: {timeframe}) after get_merged_data call."
            )

        # Ensure TIMEFRAME_MAP from simple_config is used to get lookback
        if (
            timeframe not in simple_config.TIMEFRAME_MAP
            or "lookback" not in simple_config.TIMEFRAME_MAP[timeframe]
        ):
            logger.error(
                "Timeframe '%s' or its lookback configuration not found in simple_config.TIMEFRAME_MAP.",
                timeframe,
            )
            raise ValueError(f"Invalid timeframe configuration for {timeframe}.")

        required_lookback = simple_config.TIMEFRAME_MAP[timeframe]["lookback"]

        if len(data) < required_lookback:
            logger.error(
                "Insufficient data for %s (%s): fetched %d points, but model requires lookback of %d points.",
                symbol,
                timeframe,
                len(data),
                required_lookback,
            )
            raise ValueError(
                f"Insufficient data points for {symbol} ({timeframe}) to meet model lookback requirement ({len(data)} < {required_lookback})."
            )

        logger.info(
            "Successfully fetched sufficient data for %s, timeframe %s in get_latest_data. Shape: %s, Lookback satisfied: %d >= %d",
            symbol,
            timeframe,
            data.shape,
            len(data),
            required_lookback,
        )
        return data

    except ValueError as ve:
        logger.error(
            "ValueError in get_latest_data for %s (%s): %s",
            symbol,
            timeframe,
            str(ve),
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in get_latest_data for %s (%s): %s - %s",
            symbol,
            timeframe,
            type(e).__name__,
            str(e),
            exc_info=True,
        )
        raise ValueError(
            f"Unexpected error fetching latest data for {symbol} ({timeframe}) due to {type(e).__name__}."
        ) from e


def ensemble_prediction(symbol, timeframes=["1h", "4h", "24h"]):
    """Combine predictions from multiple timeframes with dynamic weighting"""
    predictions = {}
    mape_scores = {}
    for tf in timeframes:
        result = predict_next_price(symbol, tf)
        if "error" not in result:
            predictions[tf] = result["predicted_price"]
            mape_scores[tf] = result.get("mape", 100)

    if not predictions:
        logger.error("No valid predictions for ensemble")
        return predict_next_price(symbol, "24h")

    total_inverse_mape = sum(1 / max(mape, 1e-8) for mape in mape_scores.values())
    weights = {
        tf: (1 / max(mape_scores[tf], 1e-8)) / total_inverse_mape for tf in predictions
    }
    weighted_avg = sum(predictions[tf] * weights[tf] for tf in predictions)

    last_result = predict_next_price(symbol, "24h")
    last_result["predicted_price"] = weighted_avg
    last_result["ensemble"] = predictions
    last_result["weights"] = weights
    return last_result


def fetch_market_data(symbol: str, timeframe: str):
    """Fetch market data with improved error handling and fallback mechanisms"""
    try:
        logger.info("Fetching market data for %s with timeframe %s", symbol, timeframe)
        validator = DataValidator(timeframe)

        tf_config = simple_config.TIMEFRAME_MAP.get(timeframe)
        if not tf_config:
            raise ValueError(
                f"Invalid timeframe '{timeframe}' for TIMEFRAME_MAP in fetch_market_data"
            )

        yfinance_period = tf_config.get("period")
        yfinance_interval = tf_config.get("interval")  # yfinance specific, e.g. "1d"

        if not yfinance_period or not yfinance_interval:
            raise ValueError(
                f"Missing period or interval for timeframe '{timeframe}' in TIMEFRAME_MAP"
            )

        logger.debug(
            "Attempting to fetch data from yfinance for %s using TIMEFRAME_MAP key '%s' (period: %s, interval: %s)",
            symbol, timeframe, yfinance_period, yfinance_interval
        )
        # Pass the main 'timeframe' (e.g. "24h") to fetch_yfinance_data
        primary_data = fetch_yfinance_data(
            symbol,
            period=yfinance_period,
            interval=yfinance_interval,
            timeframe=timeframe,
        )

        # Validate primary data
        if primary_data is not None and not primary_data.empty:
            primary_data, metrics = validator.validate_and_clean_data(primary_data)
            if metrics.is_valid:
                logger.info(
                    "Successfully fetched and validated %d points from primary source", len(primary_data)
                )
                return primary_data
            else:
                logger.warning("Primary data failed validation")

        # Try CoinGecko as fallback
        logger.debug("Attempting to fetch data from CoinGecko for %s", symbol)
        coingecko_data = fetch_coingecko_fallback(symbol, timeframe)

        if coingecko_data is not None and not coingecko_data.empty:
            coingecko_data, metrics = validator.validate_and_clean_data(coingecko_data)
            if metrics.is_valid:
                logger.info(
                    "Successfully fetched and validated %d points from CoinGecko", len(coingecko_data)
                )
                return coingecko_data
            else:
                logger.warning("CoinGecko data failed validation")

        # If both sources failed, try to merge them
        if primary_data is not None and coingecko_data is not None:
            logger.info("Attempting to merge data from both sources")
            merged_data = merge_data_sources(primary_data, coingecko_data)
            merged_data, metrics = validator.validate_and_clean_data(merged_data)

            if metrics.is_valid:
                logger.info(
                    "Successfully merged and validated %d points", len(merged_data)
                )
                return merged_data

        raise ValueError(
            f"Could not fetch sufficient valid data for {symbol} with timeframe {timeframe}"
        )

    except Exception as e:
        logger.error("Error fetching market data: %s", str(e))
        raise


def merge_data_sources(
    primary_data: pd.DataFrame, secondary_data: pd.DataFrame
) -> pd.DataFrame:
    """Merge data from multiple sources with careful handling of overlaps and conflicts"""
    try:
        # Ensure both dataframes have datetime index
        for df in [primary_data, secondary_data]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

        # Combine the data
        combined = pd.concat([primary_data, secondary_data])
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)

        # Handle gaps
        max_gap = pd.Timedelta(hours=24)
        gaps = combined.index[1:] - combined.index[:-1]
        large_gaps = gaps > max_gap

        if large_gaps.any():
            logger.warning("Found %d large gaps in merged data", large_gaps.sum())

        # Interpolate missing values
        combined = combined.interpolate(method="time", limit=3)

        # Validate final dataset
        if len(combined) < len(primary_data) and len(combined) < len(secondary_data):
            logger.warning("Merged dataset is smaller than both inputs")

        return combined

    except Exception as e:
        logger.error("Error merging data sources: %s", str(e))
        raise


def detect_support_resistance(
    data: pd.DataFrame,
    feature_scaler=None,
    feature_names=None,
    window: int = 20,
    num_points: int = 5,
):
    """Detect support and resistance levels using price action and volume"""
    try:
        if len(data) < window * 2 + 1:
            logger.warning(
                "Insufficient data for support/resistance detection: %d < %d",
                len(data),
                window * 2 + 1,
            )
            return {"resistance": [], "support": []}

        highs = data["High"].values
        lows = data["Low"].values
        volumes = data["Volume"].values

        # DEBUG: Log the actual price ranges to verify if data is normalized
        logger.info("[SUPPORT_RESISTANCE_DEBUG] Input data shape: %s", data.shape)
        logger.info(
            "[SUPPORT_RESISTANCE_DEBUG] High range: %.6f to %.6f",
            highs.min(),
            highs.max(),
        )
        logger.info(
            "[SUPPORT_RESISTANCE_DEBUG] Low range: %.6f to %.6f", lows.min(), lows.max()
        )
        logger.info(
            "[SUPPORT_RESISTANCE_DEBUG] Close range: %.6f to %.6f",
            data["Close"].min(),
            data["Close"].max(),
        )
        logger.info(
            "[SUPPORT_RESISTANCE_DEBUG] Sample Close values (last 5): %s",
            data["Close"].tail().values,
        )

        # Check if data is normalized and we need to denormalize
        high_is_normalized = (
            highs.max() < 100 and highs.min() > -10
        )  # Normalized data is typically in range [-3, 3]
        close_is_real = (
            data["Close"].min() > 1000
        )  # Real crypto prices are typically > $1000 for major coins

        logger.info(
            "[SUPPORT_RESISTANCE_DEBUG] High appears normalized: %s", high_is_normalized
        )
        logger.info(
            "[SUPPORT_RESISTANCE_DEBUG] Close appears real price: %s", close_is_real
        )

        # Find local maxima and minima
        resistance_points = []
        support_points = []

        logger.info(
            "[SUPPORT_RESISTANCE] Analyzing %d data points with window=%d",
            len(data),
            window,
        )

        for i in range(window, len(data) - window):
            # High points for resistance - check if current high is the highest in the window
            is_resistance = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] > highs[i]:
                    is_resistance = False
                    break

            if is_resistance:
                resistance_points.append((data.index[i], highs[i], volumes[i]))

            # Low points for support - check if current low is the lowest in the window
            is_support = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] < lows[i]:
                    is_support = False
                    break

            if is_support:
                support_points.append((data.index[i], lows[i], volumes[i]))

        logger.info(
            "[SUPPORT_RESISTANCE] Found %d resistance and %d support points",
            len(resistance_points),
            len(support_points),
        )

        # Weight points by volume and recency
        def weight_points(points):
            if not points:
                return []
            weights = []
            max_volume = max(p[2] for p in points)
            latest_time = max(p[0] for p in points)

            for point in points:
                volume_weight = point[2] / max_volume
                # Fix time calculation - use days instead of seconds for better scaling
                time_diff_days = (latest_time - point[0]).total_seconds() / 86400
                time_weight = max(
                    0.1, 1 - time_diff_days / 30
                )  # 30 days max, minimum 0.1
                total_weight = (
                    volume_weight * 0.7 + time_weight * 0.3
                )  # Weight volume more
                weights.append((point[1], total_weight))

            # Group close prices (within 2% range)
            grouped = []
            for price, weight in sorted(weights, key=lambda x: x[0]):
                added = False
                for i, (group_price, group_weight) in enumerate(grouped):
                    if abs(price - group_price) / max(group_price, 1e-8) < 0.02:
                        grouped[i] = (
                            (group_price * group_weight + price * weight)
                            / (group_weight + weight),
                            group_weight + weight,
                        )
                        added = True
                        break
                if not added:
                    grouped.append((price, weight))

            # Sort by weight and take top points
            return sorted(
                [(price, weight) for price, weight in grouped],
                key=lambda x: x[1],
                reverse=True,
            )[:num_points]

        resistance_levels = weight_points(resistance_points)
        support_levels = weight_points(support_points)

        logger.info(
            "[SUPPORT_RESISTANCE] Final levels - Resistance: %s, Support: %s",
            len(resistance_levels),
            len(support_levels),
        )

        # Denormalize support/resistance levels if they appear to be normalized
        def denormalize_levels(levels, level_type):
            if not levels:
                return levels

            denormalized_levels = []
            for price, weight in levels:
                original_price = price

                # Check if this looks like normalized data (typically in range [-3, 3])
                if -10 < price < 10 and close_is_real:
                    try:
                        # Use Close price statistics to estimate the real price scale
                        close_values = data["Close"].values
                        close_mean = np.mean(close_values)
                        close_std = np.std(close_values)

                        # For resistance levels, use High column scaling estimate
                        # For support levels, use Low column scaling estimate
                        if level_type == "resistance":
                            # Estimate High prices from Close prices (High is typically Close + some percentage)
                            high_estimate = close_mean * (
                                1 + 0.01
                            )  # Assume ~1% above close on average
                            denormalized_price = high_estimate + (
                                price * close_std * 0.1
                            )  # Scale normalized value
                        else:
                            # Estimate Low prices from Close prices (Low is typically Close - some percentage)
                            low_estimate = close_mean * (
                                1 - 0.01
                            )  # Assume ~1% below close on average
                            denormalized_price = low_estimate + (
                                price * close_std * 0.1
                            )  # Scale normalized value

                        logger.info(
                            "[SUPPORT_RESISTANCE_DENORM_V2] %s level: %.6f -> %.2f",
                            level_type,
                            original_price,
                            denormalized_price,
                        )
                        price = denormalized_price
                    except Exception as e:
                        logger.warning(
                            "[SUPPORT_RESISTANCE_DENORM_V2] Failed to denormalize %s level: %s",
                            level_type,
                            e,
                        )
                        # Keep original price if denormalization fails

                # Include any positive price; many crypto assets trade well below $1000
                if price > 0:
                    denormalized_levels.append((float(price), float(weight)))
                else:
                    logger.warning(
                        "[SUPPORT_RESISTANCE_DENORM_V2] Filtered out non-positive %s price: %s",
                        level_type,
                        price,
                    )

            return denormalized_levels

        # Apply denormalization
        resistance_levels = denormalize_levels(resistance_levels, "resistance")
        support_levels = denormalize_levels(support_levels, "support")

        logger.info(
            "[SUPPORT_RESISTANCE] After denormalization - Resistance: %s",
            resistance_levels,
        )
        logger.info(
            "[SUPPORT_RESISTANCE] After denormalization - Support: %s", support_levels
        )

        return {"resistance": resistance_levels, "support": support_levels}

    except Exception as e:
        logger.error("Error detecting support/resistance: %s", str(e), exc_info=True)
        return {"resistance": [], "support": []}


def calculate_prediction_probability(
    data: pd.DataFrame,
    prediction: float,
    model_confidence: float,
    support_resistance: dict,
    timeframe: str,
    raw_technical_indicators: dict = None,
):
    """Calculate probability metrics for the prediction"""
    try:
        current_price = data["Close"].iloc[-1]

        # Base probability from model confidence
        base_prob = float(model_confidence)  # Ensure float conversion

        # Analyze support/resistance levels with bounds checking
        price_probs = []
        for level_type in ["support", "resistance"]:
            for price, weight in support_resistance[level_type]:
                # Ensure values are float and positive
                price = float(price)
                weight = float(weight)

                # Skip invalid prices (negative or zero)
                if price <= 0:
                    continue

                # Ensure weight is positive
                weight = abs(weight)

                # Calculate proximity to S/R level
                proximity = abs(prediction - price) / max(
                    price, 1e-8
                )  # Prevent division by zero
                if proximity < 0.02:  # Within 2% of S/R level
                    signal_type = (
                        "bounce"
                        if level_type == "support" and prediction > price
                        else (
                            "reversal"
                            if level_type == "resistance" and prediction < price
                            else "breakout"
                        )
                    )

                    # Ensure confidence is positive and bounded
                    confidence = max(0, weight * (1 - proximity / 0.02))

                    price_probs.append(
                        {
                            "type": level_type,
                            "price": price,
                            "strength": weight,
                            "proximity": float(proximity),
                            "signal": signal_type,
                            "confidence": float(confidence),
                        }
                    )

        # Technical indicator probabilities
        tech_probs = []

        # RSI Analysis - Use raw values if available
        if raw_technical_indicators and "RSI_14" in raw_technical_indicators:
            rsi_value = raw_technical_indicators["RSI_14"]
            logger.info("[PREDICT_PROB] Using raw RSI value: %s", rsi_value)
        else:
            rsi_value = None
            rsi_cols = ["RSI_14", "RSI"]
            for col in rsi_cols:
                if col in data.columns:
                    rsi_value = data[col].iloc[-1]
                    # Check if RSI is scaled and convert back to 0-100 range
                    if 0 <= rsi_value <= 1:
                        rsi_value = rsi_value * 100
                        logger.info(
                            "[PREDICT_PROB] Converted scaled RSI %s to %s",
                            data[col].iloc[-1],
                            rsi_value,
                        )
                    else:
                        logger.info(
                            "[PREDICT_PROB] Using %s column for RSI analysis. Value: %s",
                            col,
                            rsi_value,
                        )
                    break

        if rsi_value is not None and 0 <= rsi_value <= 100:
            rsi_signal = {
                "indicator": "RSI",
                "value": float(rsi_value),
                "signal": (
                    "overbought"
                    if rsi_value > 70
                    else "oversold" if rsi_value < 30 else "neutral"
                ),
                "strength": (
                    0.7
                    if (rsi_value > 70 and prediction < current_price)
                    or (rsi_value < 30 and prediction > current_price)
                    else 0.3
                ),
            }
            tech_probs.append(rsi_signal)

        # MACD Analysis - Use raw values if available
        if raw_technical_indicators and "MACD" in raw_technical_indicators:
            macd = raw_technical_indicators["MACD"]
            macd_signal = raw_technical_indicators.get("MACD_Signal", 0)
            logger.info(
                "[PREDICT_PROB] Using raw MACD values: %s, signal: %s",
                macd,
                macd_signal,
            )

            trend_strength = abs(macd - macd_signal)
            macd_signal_obj = {
                "indicator": "MACD",
                "value": float(macd),
                "signal_line": float(macd_signal),
                "signal": "bullish_trend" if macd > macd_signal else "bearish_trend",
                "strength": 0.4 + min(0.3, trend_strength / 10),
                "trend_strength": float(trend_strength),
            }
            tech_probs.append(macd_signal_obj)
        elif all(col in data.columns for col in ["MACD", "MACD_Signal"]):
            macd = data["MACD"].iloc[-1]
            macd_signal = data["MACD_Signal"].iloc[-1]
            macd_prev = data["MACD"].iloc[-2] if len(data) > 1 else macd
            macd_signal_prev = (
                data["MACD_Signal"].iloc[-2] if len(data) > 1 else macd_signal
            )

            # Detect crossovers and trend strength
            crossover = (macd > macd_signal and macd_prev < macd_signal_prev) or (
                macd < macd_signal and macd_prev > macd_signal_prev
            )
            trend_strength = (
                abs(macd - macd_signal) / abs(macd_signal)
                if abs(macd_signal) > 0
                else 0
            )

            macd_signal_obj = {
                "indicator": "MACD",
                "value": float(macd),
                "signal_line": float(macd_signal),
                "signal": (
                    "bullish_crossover"
                    if macd > macd_signal and macd_prev < macd_signal_prev
                    else (
                        "bearish_crossover"
                        if macd < macd_signal and macd_prev > macd_signal_prev
                        else "bullish_trend" if macd > macd_signal else "bearish_trend"
                    )
                ),
                "strength": 0.65 if crossover else 0.4 + min(0.3, trend_strength),
                "trend_strength": float(trend_strength),
            }
            tech_probs.append(macd_signal_obj)

        # Bollinger Bands Analysis - Use raw values if available
        bb_upper = None
        bb_lower = None
        bb_middle = None

        if raw_technical_indicators and all(
            key in raw_technical_indicators
            for key in ["Bollinger_Upper", "Bollinger_Lower", "Bollinger_middle"]
        ):
            bb_upper = raw_technical_indicators["Bollinger_Upper"]
            bb_lower = raw_technical_indicators["Bollinger_Lower"]
            bb_middle = raw_technical_indicators["Bollinger_middle"]
            logger.info(
                "[PREDICT_PROB] Using raw Bollinger bands: Upper=%s, Lower=%s, Middle=%s",
                bb_upper,
                bb_lower,
                bb_middle,
            )
        else:
            bb_cols = {
                "upper": ["Bollinger_Upper", "BB_upper"],
                "lower": ["Bollinger_Lower", "BB_lower"],
                "middle": ["Bollinger_middle", "BB_middle"],
            }

            bb_values = {}
            for key, possible_cols in bb_cols.items():
                for col in possible_cols:
                    if col in data.columns:
                        bb_values[key] = data[col].iloc[-1]
                        break

            bb_upper = bb_values.get("upper", current_price)
            bb_lower = bb_values.get("lower", current_price)
            bb_middle = bb_values.get("middle", current_price)

        if (
            bb_upper is not None
            and bb_lower is not None
            and bb_middle is not None
            and bb_middle > 0
        ):
            bb_position = (
                "above_upper"
                if current_price > bb_upper
                else "below_lower" if current_price < bb_lower else "within_bands"
            )

            # Calculate bandwidth correctly (always positive)
            bandwidth = abs(bb_upper - bb_lower) / bb_middle

            bb_signal = {
                "indicator": "BollingerBands",
                "position": bb_position,
                "signal": (
                    "reversal"
                    if (bb_position == "above_upper" and prediction < current_price)
                    or (bb_position == "below_lower" and prediction > current_price)
                    else "continuation"
                ),
                "strength": 0.6 if bb_position != "within_bands" else 0.4,
                "bandwidth": float(bandwidth),
            }
            tech_probs.append(bb_signal)

        # Volume Analysis
        if "Volume" in data.columns and len(data) >= 20:
            vol_sma = data["Volume"].rolling(20).mean().iloc[-1]
            current_vol = data["Volume"].iloc[-1]
            vol_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0

            volume_signal = {
                "indicator": "Volume",
                "value": float(current_vol),
                "sma": float(vol_sma),
                "ratio": float(vol_ratio),
                "signal": (
                    "high"
                    if vol_ratio > 1.5
                    else "low" if vol_ratio < 0.5 else "normal"
                ),
                "strength": (
                    min(0.8, 0.4 + 0.2 * vol_ratio)
                    if vol_ratio > 1
                    else max(0.2, 0.4 - 0.2 * (1 - vol_ratio))
                ),
            }
            tech_probs.append(volume_signal)

        # Combine probabilities with weighted approach
        combined_prob = base_prob

        # Weight the technical probabilities
        if tech_probs:
            tech_weight = min(0.6, 0.2 * len(tech_probs))  # Cap technical weight at 0.6
            tech_prob = sum(p["strength"] for p in tech_probs) / len(tech_probs)
            combined_prob = (combined_prob * (1 - tech_weight)) + (
                tech_prob * tech_weight
            )

        # Add support/resistance influence
        if price_probs:
            sr_weight = min(0.4, 0.15 * len(price_probs))  # Cap S/R weight at 0.4
            sr_prob = sum(p["confidence"] for p in price_probs) / len(price_probs)
            combined_prob = (combined_prob * (1 - sr_weight)) + (sr_prob * sr_weight)

        # Adjust for timeframe reliability
        timeframe_factors = {
            "1m": 0.7,
            "5m": 0.75,
            "15m": 0.8,
            "30m": 0.85,
            "1h": 0.9,
            "4h": 0.95,
            "24h": 1.0,
        }
        timeframe_factor = timeframe_factors.get(timeframe, 0.85)
        combined_prob *= timeframe_factor

        # Final probability capped between 0.1 and 0.95
        final_prob = min(max(combined_prob, 0.1), 0.95)

        # Enhanced confidence factors with more detail
        confidence_factors = {
            "model_confidence": base_prob,
            "price_levels": price_probs,
            "technical_indicators": tech_probs,
            "timeframe_factor": timeframe_factor,
            "available_indicators": {
                "rsi": rsi_value is not None,
                "macd": "MACD" in data.columns and "MACD_Signal" in data.columns,
                "bollinger": bb_upper is not None
                and bb_lower is not None
                and bb_middle is not None,
                "volume": "Volume" in data.columns,
            },
            "weights": {
                "model": 1
                - (tech_weight if tech_probs else 0)
                - (sr_weight if price_probs else 0),
                "technical": tech_weight if tech_probs else 0,
                "support_resistance": sr_weight if price_probs else 0,
            },
        }

        return {"probability": final_prob, "confidence_factors": confidence_factors}

    except Exception as e:
        logger.error("Error calculating prediction probability: %s", str(e))
        # Return a more informative default response
        return {
            "probability": 0.5,  # Neutral probability
            "confidence_factors": {
                "model_confidence": model_confidence,
                "error": str(e),
                "available_columns": (
                    list(data.columns)
                    if isinstance(data, pd.DataFrame)
                    else "No DataFrame"
                ),
            },
        }


async def predict_next_price(symbol, timeframe="24h", force_retrain=False):
    """Predict the next price for a given symbol and timeframe - uses existing models by default"""
    try:
        logger.info(
            "Starting prediction for %s %s - force_retrain=%s",
            symbol,
            timeframe,
            force_retrain,
        )

        # Get model (will load existing or train new if needed)
        model, feature_scaler, target_scaler, feature_names, training_metrics = (
            await get_model(symbol, timeframe, force_retrain=force_retrain)
        )

        # Get hyperparameters for this symbol/timeframe
        timeframe_config = simple_config.TIMEFRAME_MAP.get(timeframe, {})
        lookback = timeframe_config.get("lookback", 72)

        # Model is already in eval mode from get_model
        model.eval()

        # Get latest data
        latest_data = await get_latest_data(symbol, timeframe)

        if latest_data is None or len(latest_data) < lookback:
            raise ValueError(
                f"Insufficient data points. Need at least {lookback} points."
            )

        # Store raw data for technical analysis before any scaling
        raw_data = latest_data.copy()

        # Calculate technical indicators on raw data and store before any scaling
        raw_indicators_df = calculate_technical_indicators(raw_data.copy())
        raw_technical_indicators = {
            "RSI_14": (
                float(raw_indicators_df["RSI_14"].iloc[-1])
                if "RSI_14" in raw_indicators_df.columns
                else 50.0
            ),
            "MACD": (
                float(raw_indicators_df["MACD"].iloc[-1])
                if "MACD" in raw_indicators_df.columns
                else 0.0
            ),
            "MACD_Signal": (
                float(raw_indicators_df["MACD_Signal"].iloc[-1])
                if "MACD_Signal" in raw_indicators_df.columns
                else 0.0
            ),
            "Bollinger_Upper": (
                float(raw_indicators_df["Bollinger_Upper"].iloc[-1])
                if "Bollinger_Upper" in raw_indicators_df.columns
                else 0.0
            ),
            "Bollinger_Lower": (
                float(raw_indicators_df["Bollinger_Lower"].iloc[-1])
                if "Bollinger_Lower" in raw_indicators_df.columns
                else 0.0
            ),
            "Bollinger_middle": (
                float(raw_indicators_df["Bollinger_middle"].iloc[-1])
                if "Bollinger_middle" in raw_indicators_df.columns
                else 0.0
            ),
        }
        logger.info(
            "[PREDICT_NEXT_PRICE] Raw technical indicators: %s", raw_technical_indicators
        )

        # Prepare features using the same feature names from training
        features_df, _ = await prepare_features(symbol, timeframe)
        features = features_df[feature_names].values

        # Scale features
        scaled_features = feature_scaler.transform(features)

        # Create sequence for prediction (take last lookback points)
        x_data = scaled_features[-lookback:].reshape(
            1, lookback, -1
        )  # Let reshape infer the last dimension

        # Convert to tensor and make prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_data).to(device)
            prediction = model(x_tensor)
            prediction = prediction.cpu().numpy()

        # Inverse transform prediction
        prediction_denorm = target_scaler.inverse_transform(prediction.reshape(-1, 1))
        predicted_value = float(prediction_denorm[-1, 0])

        # Get last actual price for comparison
        last_price = float(latest_data["Close"].iloc[-1])

        # Convert prediction based on target type
        if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
            # For short timeframes, prediction is percentage change
            predicted_price = last_price * (1 + predicted_value / 100)
            logger.info(
                "[PREDICT_NEXT_PRICE] Converted percentage change %.4f%% to price: %.2f -> %.2f",
                predicted_value,
                last_price,
                predicted_price,
            )
        else:
            # For longer timeframes, prediction is absolute price
            predicted_price = predicted_value
            logger.info(
                "[PREDICT_NEXT_PRICE] Using absolute price prediction: %.2f",
                predicted_price,
            )

        # Get truly raw data (before any scaling) for support/resistance detection
        try:
            data_fetcher = DataFetcher()
            # Get raw data without any technical indicators or scaling
            raw_unscaled_data = await data_fetcher.get_merged_data(symbol, timeframe)
            if raw_unscaled_data is None or len(raw_unscaled_data) < 20:
                logger.warning(
                    "Could not fetch raw unscaled data, using processed data"
                )
                raw_unscaled_data = raw_data
        except Exception as e:
            logger.warning(
                "Error fetching raw unscaled data: %s, using processed data", e
            )
            raw_unscaled_data = raw_data

        # Calculate confidence metrics using truly raw data for support/resistance
        support_resistance = detect_support_resistance(
            raw_unscaled_data, feature_scaler, feature_names
        )
        probability = calculate_prediction_probability(
            raw_data,  # Use raw data for price analysis
            predicted_price,
            0.8,  # Model confidence
            support_resistance,
            timeframe,
            raw_technical_indicators,  # Pass raw technical indicators
        )

        # Get next timestamp
        next_timestamp = get_next_timestamp(latest_data.index[-1], timeframe)

        # Use the metrics returned from get_model
        final_metrics = training_metrics if training_metrics else {}

        # Format support/resistance levels for API response
        price_levels = []

        # Add resistance levels
        for price, weight in support_resistance.get("resistance", []):
            price_levels.append(
                {"type": "resistance", "price": float(price), "strength": float(weight)}
            )

        # Add support levels
        for price, weight in support_resistance.get("support", []):
            price_levels.append(
                {"type": "support", "price": float(price), "strength": float(weight)}
            )

        logger.info(
            "[PREDICT_NEXT_PRICE] Formatted price_levels for API: %s", price_levels
        )

        # Prepare enhanced result with rich model performance metrics
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "predicted_price": predicted_price,
            "last_price": last_price,
            "prediction_time": next_timestamp.isoformat(),
            "current_time": datetime.now(pytz.UTC).isoformat(),
            "probability": probability.get("probability", 0.5),
            "confidence_factors": probability.get("confidence_factors", {}),
            "price_levels": price_levels,  # Add the formatted support/resistance levels
            "model_performance": {
                "r2_score": final_metrics.get("r2", final_metrics.get("val_r2", 0.0)),
                "mape": final_metrics.get("mape", final_metrics.get("val_mape", 0.0)),
                "mae": final_metrics.get("mae", final_metrics.get("val_mae", 0.0)),
                "rmse": final_metrics.get("rmse", final_metrics.get("val_rmse", 0.0)),
                "validation_loss": final_metrics.get("val_loss", 0.0),
                "trained_on_demand": final_metrics.get(
                    "trained_on_demand", force_retrain
                ),
                "best_epoch": final_metrics.get("best_epoch", 0),
                "model_architecture": {
                    "lookback": lookback,
                    "input_size": len(feature_names),
                    "hidden_size": timeframe_config.get("hidden_size", 128),
                    "num_layers": timeframe_config.get("num_layers", 2),
                    "dropout": timeframe_config.get("dropout", 0.3),
                },
            },
        }

        logger.info("Prediction result for %s %s: %s", symbol, timeframe, result)
        return result

    except Exception as e:
        logger.error("Error predicting price for %s %s: %s", symbol, timeframe, str(e))
        raise


async def prepare_features(symbol: str, timeframe: str) -> tuple:
    """Prepare features for model input with proper shapes"""
    try:
        # Get historical data
        data_fetcher = DataFetcher()
        df = await data_fetcher.get_merged_data(symbol, timeframe)

        if df is None or df.empty:
            logger.error("Failed to fetch data for %s %s", symbol, timeframe)
            raise ValueError(f"No data available for {symbol} {timeframe}")

        # Forward fill missing values, then backward fill any remaining
        df = df.ffill().bfill()

        # Generate technical indicators
        df_with_ta = data_fetcher._add_technical_indicators(df.copy())
        feature_names = df_with_ta.columns.tolist()

        if df_with_ta is None or df_with_ta.empty:
            logger.error("Failed to generate features for %s %s", symbol, timeframe)
            raise ValueError(f"Feature generation failed for {symbol} {timeframe}")

        # Ensure all required columns are present
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [
            col for col in required_columns if col not in df_with_ta.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Create target series based on timeframe
        # For shorter timeframes (≤1h), predict percentage change to avoid unrealistic volatility
        # For longer timeframes (>1h), predict absolute price
        if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
            # Predict percentage change for short timeframes
            target_series = (
                df_with_ta["Close"].pct_change(periods=1).shift(-1) * 100
            )  # Convert to percentage
            logger.info("Using percentage change target for %s timeframe", timeframe)
        else:
            # Predict absolute price for longer timeframes
            target_series = df_with_ta["Close"].shift(-1)
            logger.info("Using absolute price target for %s timeframe", timeframe)

        # Drop the last row since it won't have a target value
        df_with_ta = df_with_ta[:-1]
        target_series = target_series[:-1]

        # Handle any remaining NaN values
        if df_with_ta.isna().any().any():
            logger.warning(
                "NaN values found in features, filling with forward fill then backward fill"
            )
            df_with_ta = df_with_ta.ffill().bfill()

        if target_series.isna().any():
            logger.warning(
                "NaN values found in target series, filling with forward fill then backward fill"
            )
            target_series = target_series.ffill().bfill()

        # Log shapes for debugging
        logger.debug("Features shape: %s", df_with_ta.shape)
        logger.debug("Target shape: %s", target_series.shape)

        # CRITICAL FIX: Drop rows with NaNs introduced by feature engineering (e.g., momentum, lag)
        # Align features and target by index after dropping NaNs
        features_df = df_with_ta.dropna()
        target_series = target_series[features_df.index]

        # --- Stage 2 Feature Selection ---
        pre_feature_count = features_df.shape[1]
        # Stage 2: statistical feature selection
        features_df = _select_important_features(features_df, target_series)

        # Stage 3: outlier clipping to tame extreme values
        features_df = _clip_outliers(features_df)
        logger.info(
            "Feature selection reduced columns from %d to %d",
            pre_feature_count,
            features_df.shape[1],
        )

        if features_df.empty:
            raise ValueError(
                "No data left after cleaning NaNs. Check feature generation."
            )

        return features_df, target_series

    except Exception as e:
        logger.error("Error preparing features: %s", str(e))
        raise


def _select_important_features(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    corr_threshold: float = 0.05,
    variance_threshold: float = 1e-8,
    multicollinearity_threshold: float = 0.95,
    mutual_info_threshold: float = 0.001,
) -> pd.DataFrame:
    """Advanced statistical feature filtering.

    Steps performed:
    1. Remove low-variance features (variance < ``variance_threshold``).
    2. Remove features with low absolute Pearson correlation to the target (|corr| < ``corr_threshold``).
    3. Remove one feature from highly collinear feature pairs (|corr| > ``multicollinearity_threshold``).
    4. Remove features with low mutual information with the target (MI < ``mutual_info_threshold``).

    Core OHLCV columns (``Open``, ``High``, ``Low``, ``Close``, ``Volume``) are never dropped.
    """
    try:
        core_cols = {"Open", "High", "Low", "Close", "Volume"}
        numeric_df = features_df.select_dtypes(include=[np.number])

        # 1. Low variance
        low_variance_cols = numeric_df.var().loc[lambda s: s < variance_threshold].index

        # 2. Low Pearson correlation
        corr_with_target = numeric_df.corrwith(target_series).abs()
        low_corr_cols = corr_with_target.loc[lambda s: s < corr_threshold].index

        # 3. Multicollinearity
        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        )
        high_corr_cols = [
            col
            for col in upper_triangle.columns
            if any(upper_triangle[col] > multicollinearity_threshold)
        ]

        # 4. Mutual information (non-linear dependence)
        try:
            mi_scores = mutual_info_regression(
                numeric_df.values, target_series.values, random_state=42
            )
            mi_series = pd.Series(mi_scores, index=numeric_df.columns)
            low_mi_cols = mi_series.loc[mi_series < mutual_info_threshold].index
        except Exception as mi_err:
            logger.warning(
                "[FEATURE_SELECTION] mutual_info_regression failed – skipping MI filter: %s",
                mi_err,
            )
            low_mi_cols = []

        # Consolidate columns to drop (excluding core OHLCV)
        cols_to_drop = (
            set(low_variance_cols)
            | set(low_corr_cols)
            | set(high_corr_cols)
            | set(low_mi_cols)
        ) - core_cols

        if cols_to_drop:
            logger.info(
                "[FEATURE_SELECTION] Dropping %d features | low_var=%d, low_corr=%d, multi_collinear=%d, low_mi=%d: %s",
                len(cols_to_drop),
                len(low_variance_cols),
                len(low_corr_cols),
                len(high_corr_cols),
                len(low_mi_cols),
                list(cols_to_drop),
            )
            features_df = features_df.drop(columns=list(cols_to_drop))
        else:
            logger.info("[FEATURE_SELECTION] No features dropped (all passed thresholds)")

        return features_df

    except Exception as e:
        logger.error("[FEATURE_SELECTION] Error selecting features: %s", str(e), exc_info=True)
        return features_df


def _clip_outliers(features_df: pd.DataFrame, z_thresh: float = 5.0) -> pd.DataFrame:
    """Clip extreme outliers for numeric columns based on z-score threshold.

    Any value whose absolute z-score exceeds ``z_thresh`` standard deviations
    from the mean is clipped to the corresponding boundary. This mitigates the
    influence of extreme spikes while preserving overall data distribution.
    """
    try:
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return features_df

        clipped_total = 0
        for col in numeric_cols:
            series = features_df[col]
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
            lower = mean - z_thresh * std
            upper = mean + z_thresh * std
            before = series.copy()
            features_df[col] = series.clip(lower=lower, upper=upper)
            clipped_total += (before != features_df[col]).sum()

        if clipped_total > 0:
            logger.info(
                "[OUTLIER_CLIP] Clipped %d extreme values (> %.1fσ)",
                clipped_total,
                z_thresh,
            )
        return features_df
    except Exception as e:
        logger.error("[OUTLIER_CLIP] Error during clipping: %s", str(e), exc_info=True)
        return features_df

    """Select a subset of informative features.

    This function performs three simple but effective checks:
    1. Low variance: drops features whose variance is below `variance_threshold`.
    2. Low correlation: drops features whose absolute Pearson correlation with the target is below `corr_threshold`.
    3. Multicollinearity: for any pair of features with absolute correlation above
       `multicollinearity_threshold`, the later-occurring feature is dropped.

    Core OHLCV columns are **never** removed.
    """
    try:
        core_cols = {"Open", "High", "Low", "Close", "Volume"}

        # Work only on numeric data for correlation / variance checks
        numeric_df = features_df.select_dtypes(include=[np.number])

        # 1. Low variance
        low_variance_cols = numeric_df.var()[numeric_df.var() < variance_threshold].index

        # 2. Correlation with target
        corr_with_target = numeric_df.corrwith(target_series).abs()
        low_corr_cols = corr_with_target[corr_with_target < corr_threshold].index

        # 3. Multicollinearity between features
        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
        high_corr_cols = [
            col for col in upper_triangle.columns if any(upper_triangle[col] > multicollinearity_threshold)
        ]

        cols_to_drop = (
            set(low_variance_cols) | set(low_corr_cols) | set(high_corr_cols)
        ) - core_cols

        if cols_to_drop:
            logger.info(
                "[FEATURE_SELECTION] Dropping %d low-importance features: %s",
                len(cols_to_drop),
                list(cols_to_drop),
            )
            features_df = features_df.drop(columns=list(cols_to_drop))

        return features_df

    except Exception as e:
        logger.error("[FEATURE_SELECTION] Error selecting features: %s", str(e))
        # In case of any error, fall back to original features
        return features_df


def check_data_drift(old_data, new_data, feature_names):
    """Detect data drift using KS test with multiple features"""
    if len(old_data) < 10 or len(new_data) < 10:
        return True
    # Only check first 8 features for efficiency, or all if less than 8
    features_to_check = feature_names[: min(8, len(feature_names))]
    for i, feature in enumerate(features_to_check):
        _, p = ks_2samp(old_data[:, i], new_data[:, i])
        if p < 0.01:
            logger.info("Data drift detected in feature %s", feature)
            return True
    return False


async def get_model(symbol, timeframe="24h", lookback=None, force_retrain=False):
    """Get or train model for given symbol and timeframe using robust ModelManager.

    Args:
        force_retrain: If True, delete existing model and train fresh (default: False).
    """
    try:
        timeframe_config = simple_config.TIMEFRAME_MAP.get(timeframe, {})
        if lookback is None:
            lookback = timeframe_config.get("lookback", 72)



        (
            x_train,
            y_train,
            x_val,
            y_val,
            feature_scaler,
            target_scaler,
            feature_names,
            _,
        ) = await _get_and_prepare_training_data(symbol, timeframe, lookback)

        input_size = x_train.shape[2]
        hidden_size = timeframe_config.get("hidden_size", 128)
        num_layers = timeframe_config.get("num_layers", 2)
        dropout = timeframe_config.get("dropout", 0.3)

        expected_config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_list": feature_names,
            "lookback": lookback,
        }

        if force_retrain:
            logger.info(
                "Force retrain for %s %s, deleting existing model.", symbol, timeframe
            )
            model_manager.delete_model(symbol, timeframe)
        else:
            loaded_model, loaded_fs, loaded_ts, _, _ = (
                model_manager.load_model_and_scalers(symbol, timeframe)
            )
            if loaded_model and model_manager.check_model_compatibility(
                symbol, timeframe, expected_config
            ):
                logger.info(
                    "Using existing compatible model for %s %s.", symbol, timeframe
                )
                metrics = model_manager.get_latest_metrics(symbol, timeframe) or {}
                loaded_model.eval()
                return loaded_model, loaded_fs, loaded_ts, feature_names, metrics
            if loaded_model:
                logger.warning(
                    "Model config mismatch for %s %s, retraining.", symbol, timeframe
                )
                model_manager.delete_model(symbol, timeframe)

        logger.info("Training new model for %s %s.", symbol, timeframe)
        model = BiLSTMWithAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        model, training_history = await train_model(
            x_train,
            y_train,
            x_val,
            y_val,
            model,
            timeframe,
            symbol,
            target_scaler,
        )

        training_metrics = training_history.get("best_metrics", {})
        training_metrics["trained_on_demand"] = True

        training_info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "training_date": datetime.now(pytz.UTC).isoformat(),
            "data_points": len(x_train) + len(x_val),
            "training_history": training_history,
        }

        success = model_manager.save_model_and_scalers(
            model,
            feature_scaler,
            target_scaler,
            symbol,
            timeframe,
            expected_config,
            training_info,
            training_metrics,
        )

        if success:
            logger.info(
                "Successfully saved new model for %s %s with metrics: %s",
                symbol,
                timeframe,
                training_metrics,
            )
        else:
            logger.error("Failed to save new model for %s %s", symbol, timeframe)
            raise IOError(f"Could not save model for {symbol} {timeframe}")

        model.eval()
        return model, feature_scaler, target_scaler, feature_names, training_metrics

    except Exception as e:
        logger.error(
            "Error in get_model for %s %s: %s", symbol, timeframe, e, exc_info=True
        )
        raise


async def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    model,
    timeframe,
    symbol,
    target_scaler,
    hyperparams: Optional[Dict] = None,
):
    """Train the model with AdamW, HuberLoss, and OneCycleLR scheduler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get hyperparameters from config
    timeframe_config = simple_config.TIMEFRAME_MAP.get(timeframe, {})
    batch_size = timeframe_config.get("batch_size", 64)
    learning_rate = timeframe_config.get("learning_rate", 0.001)
    max_lr = timeframe_config.get("max_lr", 0.01)
    epochs = timeframe_config.get("epochs", 100)
    patience = timeframe_config.get("patience", 15)

    # Override default training hyperparameters with values provided via `hyperparams`
    if hyperparams:
        batch_size = hyperparams.get("batch_size", batch_size)
        learning_rate = hyperparams.get("learning_rate", learning_rate)
        max_lr = hyperparams.get("max_lr", max_lr)
        epochs = hyperparams.get("epochs", epochs)
        patience = hyperparams.get("patience", patience)

    logger.info(
        "Starting training for %s %s with config: "
        "epochs=%s, batch_size=%s, initial_lr=%s, max_lr=%s, patience=%s",
        symbol,
        timeframe,
        epochs,
        batch_size,
        learning_rate,
        max_lr,
        patience,
    )

    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer, loss, and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        # Validation loop
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # Denormalize for metric calculation
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        preds_denorm = target_scaler.inverse_transform(all_preds)
        targets_denorm = target_scaler.inverse_transform(all_targets)

        # Calculate metrics with special handling for percentage-change targets.
        # For short intraday timeframes our targets are percentage changes (in pct-points).
        # Classical MAPE is unstable when targets are close to zero, so we skip it and
        # rely on MAE / RMSE instead.
        val_r2 = r2_score(targets_denorm, preds_denorm)

        percent_change_tfs = ["1m", "5m", "15m", "30m", "1h"]
        if timeframe in percent_change_tfs:
            # Targets represent pct-change → treat MAE/RMSE as pct-point errors.
            val_mae = mean_absolute_error(targets_denorm, preds_denorm)
            val_rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
            val_mape = np.nan  # Not meaningful for pct-targets
        else:
            val_mae = mean_absolute_error(targets_denorm, preds_denorm)
            val_rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
            val_mape = (
                np.mean(
                    np.abs(
                        (targets_denorm - preds_denorm)
                        / np.where(targets_denorm == 0, 1e-6, targets_denorm)
                    )
                )
                * 100
            )

        logger.info(
            "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | R²: %.4f | MAPE: %.2f%%",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_r2,
            val_mape,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            best_metrics = {
                "val_loss": best_val_loss,
                "val_r2": val_r2,
                "val_mae": val_mae,
                "val_mape": val_mape,
                "val_rmse": val_rmse,
                "best_epoch": epoch + 1,
            }
            logger.info(
                "New best model found at epoch %d with Val Loss: %.6f",
                epoch + 1,
                best_val_loss,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                "Early stopping triggered after %d epochs with no improvement.", patience
            )
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    training_summary = {"history": history, "best_metrics": best_metrics}
    return model, training_summary


def tune_model_optuna(
    symbol: str,
    timeframe: str = "24h",
    lookback: Optional[int] = None,
    n_trials: int = 25,
    forecast_horizon: int = 1,
    scaler_type: str = "standard",
) -> dict:
    """Run Optuna hyper-parameter search to optimise BiLSTMWithAttention.

    This helper prepares the training data **once** and evaluates different
    architecture / optimiser settings across *n_trials* Optuna trials.

    Returns a dictionary containing the best parameters and the corresponding
    validation loss. Requires the *optuna* package (``pip install optuna``).
    """
    if optuna is None:
        raise ImportError(
            "optuna package is required for hyperparameter tuning. Install it via `pip install optuna`."
        )

    # Prepare data exactly once outside the objective to speed up tuning
    async def _prepare():
        _lookback = lookback or simple_config.TIMEFRAME_MAP.get(timeframe, {}).get("lookback", 72)
        return await _get_and_prepare_training_data(
            symbol,
            timeframe,
            _lookback,
            forecast_horizon=forecast_horizon,
            scaler_type=scaler_type,
        )

    X_train, y_train, X_val, y_val, feature_scaler, target_scaler, feature_names, _ = asyncio.run(_prepare())
    input_size = X_train.shape[2]

    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        max_lr = trial.suggest_loguniform("max_lr", 5e-4, 5e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = BiLSTMWithAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        hyperparams = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_lr": max_lr,
            "epochs": 60,
            "patience": 10,
        }

        # Train and obtain validation loss
        _, summary = asyncio.run(
            train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                model,
                timeframe,
                symbol,
                target_scaler,
                hyperparams=hyperparams,
            )
        )
        return summary["best_metrics"]["val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return {"best_params": study.best_trial.params, "best_val_loss": study.best_value}


class DataAugmentor:
    """Advanced feature engineering and data augmentation"""

    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.feature_importance = {}

    def add_price_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        try:
            df = data.copy()

            # Basic candlestick features
            df["body_size"] = df["Close"] - df["Open"]
            df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
            df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
            df["body_to_shadow_ratio"] = df["body_size"].abs() / (
                df["upper_shadow"] + df["lower_shadow"] + 1e-8
            )

            # Candlestick patterns
            df["doji"] = (
                abs(df["body_size"]) <= 0.1 * (df["High"] - df["Low"])
            ).astype(float)
            df["hammer"] = (
                (df["lower_shadow"] > 2 * abs(df["body_size"]))
                & (df["upper_shadow"] <= 0.2 * df["lower_shadow"])
            ).astype(float)
            df["shooting_star"] = (
                (df["upper_shadow"] > 2 * abs(df["body_size"]))
                & (df["lower_shadow"] <= 0.2 * df["upper_shadow"])
            ).astype(float)

            return df

        except Exception as e:
            logger.error("Error adding price patterns: %s", str(e))
            return data

    def add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility features"""
        try:
            df = data.copy()

            # Parkinson volatility
            df["parkinson_vol"] = np.sqrt(
                1 / (4 * np.log(2)) * ((df["High"] / df["Low"]).apply(np.log)) ** 2
            )

            # Garman-Klass volatility
            df["garman_klass_vol"] = np.sqrt(
                0.5 * np.log(df["High"] / df["Low"]) ** 2
                - (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"]) ** 2
            )

            # Rolling volatility measures
            windows = [5, 10, 20] if self.timeframe in ["30m", "1h"] else [3, 5, 10]

            for window in windows:
                # Standard deviation of returns
                df[f"return_vol_{window}"] = (
                    df["Close"].pct_change().rolling(window).std()
                )

                # Range-based volatility
                df[f"range_vol_{window}"] = (
                    ((df["High"] - df["Low"]) / df["Close"]).rolling(window).mean()
                )

                # Realized volatility
                df[f"realized_vol_{window}"] = np.sqrt(
                    (df["Close"].pct_change() ** 2).rolling(window).sum() / window
                )

            return df

        except Exception as e:
            logger.error("Error adding volatility features: %s", str(e))
            return data

    def add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and trend features"""
        try:
            df = data.copy()

            # Define lookback periods based on timeframe
            if self.timeframe in ["30m", "1h"]:
                periods = [12, 24, 48, 96]  # 6h, 12h, 24h, 48h for hourly data
            else:
                periods = [3, 5, 10, 20]

            for period in periods:
                # ROC (Rate of Change)
                df[f"roc_{period}"] = df["Close"].pct_change(period)

                # Momentum
                df[f"momentum_{period}"] = df["Close"] - df["Close"].shift(period)

                # Acceleration
                df[f"acceleration_{period}"] = df[f"momentum_{period}"] - df[
                    f"momentum_{period}"
                ].shift(1)

                # Trend strength
                df[f"trend_strength_{period}"] = (
                    df["Close"].rolling(period).mean() / df["Close"] - 1
                )

            # Add RSI variations
            df["rsi_smooth"] = df["RSI"].rolling(3).mean()
            df["rsi_impulse"] = df["RSI"] - df["RSI"].shift(3)

            return df

        except Exception as e:
            logger.error("Error adding momentum features: %s", str(e))
            return data

    def add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volume analysis features"""
        try:
            df = data.copy()

            # Volume momentum
            df["volume_momentum"] = df["Volume"].pct_change()

            # Volume weighted average price variations
            windows = [5, 10, 20] if self.timeframe in ["30m", "1h"] else [3, 5, 10]

            for window in windows:
                # VWAP
                df[f"vwap_{window}"] = (
                    df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
                ).rolling(window).sum() / df["Volume"].rolling(window).sum()

                # Volume force
                df[f"volume_force_{window}"] = (
                    df["volume_momentum"] * df["Close"].pct_change()
                )

                # Price-volume trend
                df[f"pvt_{window}"] = (
                    (df["Close"].pct_change() * df["Volume"]).rolling(window).sum()
                )

            # Volume profile
            df["volume_price_spread"] = (df["High"] - df["Low"]) * df["Volume"]
            df["volume_price_correlation"] = df["Close"].rolling(20).corr(df["Volume"])

            return df

        except Exception as e:
            logger.error("Error adding volume features: %s", str(e))
            return data

    def generate_synthetic_samples(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic samples for rare events"""
        try:
            df = data.copy()

            # Identify rare events (e.g., large price movements)
            returns = df["Close"].pct_change()
            std_dev = returns.std()
            rare_events = abs(returns) > 2 * std_dev  # More than 2 standard deviations

            if rare_events.sum() < len(df) * 0.1:  # Less than 10% rare events
                logger.info("Generating synthetic samples for rare events")

                rare_samples = df[rare_events].copy()

                # Create variations of rare events
                for _ in range(3):  # Generate 3 variations for each rare event
                    variation = rare_samples.copy()

                    # Add small random variations to features
                    for col in variation.select_dtypes(include=[np.number]).columns:
                        noise = np.random.normal(0, 0.01, len(variation))
                        variation[col] = variation[col] * (1 + noise)

                    df = pd.concat([df, variation])

            return df.sort_index()

        except Exception as e:
            logger.error("Error generating synthetic samples: %s", str(e))
            return data

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering and augmentation steps"""
        try:
            df = data.copy()

            # 1. Add price patterns
            df = self.add_price_patterns(df)

            # 2. Add volatility features
            df = self.add_volatility_features(df)

            # 3. Add momentum features
            df = self.add_momentum_features(df)

            # 4. Add volume features
            df = self.add_volume_features(df)

            # 5. Generate synthetic samples if needed
            if len(df) > 100:  # Only for sufficient data
                df = self.generate_synthetic_samples(df)

            # Remove any features with too many missing values
            missing_pct = df.isnull().sum() / len(df)
            cols_to_drop = missing_pct[missing_pct > 0.1].index
            if not cols_to_drop.empty:
                logger.warning(
                    "Dropping columns with too many missing values: %s",
                    cols_to_drop.tolist(),
                )
                df = df.drop(columns=cols_to_drop)

            # Fill remaining missing values
            df = df.ffill().bfill()

            return df

        except Exception as e:
            logger.error("Error in augmentation pipeline: %s", str(e))
            raise


def main(symbol: str, timeframe: str):
    """Command-line entry point for prediction"""
    try:
        logger.debug(
            "Script started: Predicting for %s with timeframe %s", symbol, timeframe
        )
        print(f"Starting prediction for {symbol} with timeframe {timeframe}")
        if timeframe not in simple_config.TIMEFRAME_OPTIONS:
            error_msg = f"Invalid timeframe. Must be one of: {', '.join(simple_config.TIMEFRAME_OPTIONS)}"
            logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
        if timeframe == "24h":
            result = ensemble_prediction(symbol)
        else:
            result = predict_next_price(symbol, timeframe)
        if "error" in result:
            logger.error("Prediction error: %s", result['error'])
            print(f"Error: {result['error']}")
            sys.exit(1)
        logger.debug("Prediction result: %s", result)
        print(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m controllers.predictor <symbol> <timeframe>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
