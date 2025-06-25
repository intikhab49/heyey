"""Module for fetching and processing cryptocurrency market data."""

import asyncio
import os
import logging
import time
from functools import wraps
from typing import Optional, Dict, Any, List

import aiohttp
import pandas as pd
import yfinance as yf
from cachetools import TTLCache, cached
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
)

import simple_config
import numpy as np
import ta
import collections
from pathlib import Path



# 1. Custom Exception for graceful failure
class DataUnavailableError(Exception):
    """Custom exception raised when data cannot be fetched from any source."""



# Cache for CoinGecko ID lookups
coin_id_cache = TTLCache(maxsize=1000, ttl=86400)

# 2. Intelligent Caching for API responses
# Cache holds 50 items, expires after 10 minutes (600 seconds)
api_cache = TTLCache(maxsize=50, ttl=600)

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
logger.setLevel(logging.INFO)

# ---------------- Global Rate-Limit & Cache Settings ---------------- #
# Local on-disk cache directory for raw OHLCV frames
DATA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Per-endpoint conservative rate limits (req, seconds)
RATE_LIMITS = {
    "cg_free": (50, 60),    # CoinGecko free API (max 50/min)
    "cg_demo": (30, 60),     # CoinGecko demo API (demo key plan)
    "yfinance": (120, 60),  # yfinance scraping (empirical safe limit)
    "alpha_vantage": (5, 60),  # Alpha Vantage free tier
}

def endpoint_rate_limiter(name: str):
    """Return a decorator enforcing the specified endpoint's limits."""
    max_calls, period = RATE_LIMITS[name]
    return async_rate_limiter(max_calls, period)


# 3. Intelligent Rate Limiter (Asynchronous)
def async_rate_limiter(max_calls: int, period: int):
    """Ensures a function is not called more than `max_calls` times in a `period` of seconds."""
    calls = []
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.monotonic()
            calls[:] = [c for c in calls if c > now - period]
            if len(calls) >= max_calls:
                wait_time = (calls[0] + period) - now
                logger.warning(
                "Rate limit proactively applied. Waiting for %.2f seconds.", wait_time
            )
                await asyncio.sleep(wait_time)

            # Add the current call's timestamp *before* the call
            calls.append(time.monotonic())
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Define what exceptions are considered for retries
RETRYABLE_EXCEPTIONS = (
    aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ClientResponseError
)


class DataFetcher:
    """Manages fetching data from various sources like CoinGecko and yfinance."""
    def __init__(self):
        self.alpha_vantage_ts_client = None # Alpha Vantage remains disabled as per original logic
        self.alpha_vantage_cc_client = None
        # Read optional CoinGecko Demo API key from environment
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY", "")

        # ---------------- Disk Cache Helpers ---------------- #
    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return DATA_CACHE_DIR / f"{symbol.lower()}_{timeframe}.parquet"

    def _load_cached(self, symbol: str, timeframe: str):
        path = self._cache_path(symbol, timeframe)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Failed to read cache %s: %s", path, exc)
        return None

    def _save_cached(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol, timeframe)
        try:
            df.to_parquet(path, compression='snappy')
        except Exception as exc:
            logger.warning("Failed to write cache %s: %s", path, exc)

    # 4. Solidified Fallback Logic with Caching
    async def get_merged_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetches data by trying CoinGecko, then yfinance.
        Applies caching, rate limiting, and exponential backoff.
        Raises DataUnavailableError if all sources fail.
        """
        logger.info("Fetching merged data for %s (%s)...", symbol, timeframe)
        df = None
        source_used = None

        # ---------------- Check Local Disk Cache First ---------------- #
        min_required = simple_config.TIMEFRAME_MAP[timeframe]['lookback'] * 4
        cached_df = await asyncio.to_thread(self._load_cached, symbol, timeframe)
        if cached_df is not None and len(cached_df) >= min_required:
            logger.info("Loaded %d rows from disk cache for %s (%s)", len(cached_df), symbol, timeframe)
            return cached_df

        # 1. Try CoinGecko
        try:
            df = await self.get_coingecko_data(symbol, timeframe)
            if df is not None and not df.empty:
                source_used = "CoinGecko"
                logger.info("Successfully fetched data for %s from CoinGecko.", symbol)
        except RetryError as e:
            logger.warning("CoinGecko fetch failed for %s after all retries: %s", symbol, e)
        except (aiohttp.ClientError, ValueError, KeyError) as e:
            logger.error("Unexpected CoinGecko fetch error for %s: %s", symbol, e)

        # 2. Try yfinance if CoinGecko fails
        if df is None or df.empty:
            logger.info("CoinGecko failed for %s. Trying yfinance.", symbol)
            try:
                df = await self.fetch_yfinance_data(symbol, timeframe)
                if df is not None and not df.empty:
                    source_used = "yfinance"
                    # Flatten MultiIndex columns from yfinance and capitalize
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    df.columns = [str(col).capitalize() for col in df.columns]
                    logger.info("Successfully fetched data for %s from yfinance.", symbol)
            except RetryError as e:
                logger.warning("yfinance fetch failed for %s after all retries: %s", symbol, e)
            except (IOError, ValueError, KeyError) as e:
                logger.error("Unexpected yfinance fetch error for %s: %s", symbol, e)

            # 3. Try Alpha Vantage as final fallback
            if df is None or df.empty:
                logger.info("CoinGecko and yfinance failed for %s. Trying Alpha Vantage.", symbol)
                try:
                    df = await self.fetch_alpha_vantage_data(symbol, timeframe)
                    if df is not None and not df.empty:
                        source_used = "Alpha Vantage"
                        logger.info("Successfully fetched data for %s from Alpha Vantage.", symbol)
                except Exception as av_exc:
                    logger.error("Alpha Vantage fetch error for %s: %s", symbol, av_exc)

        if df is not None and not df.empty:
            log_msg = "Data fetched for %s from %s. Shape: %s"
            logger.info(log_msg, symbol, source_used, df.shape)
            # Persist to local cache asynchronously
            await asyncio.to_thread(self._save_cached, symbol, timeframe, df)
            return df

        logger.error("All data sources failed for %s (%s).", symbol, timeframe)
        raise DataUnavailableError(
            f"Could not retrieve sufficient data for {symbol} ({timeframe})."
        )

    # 5. Refactored Core Functions with Exponential Backoff
    @async_rate_limiter(max_calls=30, period=60)  # CoinGecko (free/demo) tier

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True
    )
    async def get_coingecko_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetches data from CoinGecko with rate limiting and exponential backoff."""
        logger.debug("Attempting to fetch CoinGecko data for %s (%s)", symbol, timeframe)
        coin_id = await self.get_coingecko_id(symbol)
        if not coin_id:
            return None

        # Use the period from the definitive TIMEFRAME_MAP
        period_str = simple_config.TIMEFRAME_MAP[timeframe].get('period', '30d')
        # Extract numeric part of the period string (e.g., '30d' -> 30)
        days = int("".join(filter(str.isdigit, period_str)))

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': str(days)}
        headers = {"x-cg-demo-api-key": self.coingecko_api_key} if self.coingecko_api_key else None

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=20) as response:
                if response.status == 429:
                    logger.warning("CoinGecko rate limit hit (429). Waiting for 60s before retry.")
                    await asyncio.sleep(60)
                    response.raise_for_status() # Trigger a retry after waiting

                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = await response.json()
                df = self._process_coingecko_data(data, timeframe)

                # Dynamically calculate minimum required samples
                min_required = simple_config.TIMEFRAME_MAP[timeframe]['lookback'] * 4
                if df is None or df.empty or len(df) < min_required:
                    logger.warning(
                    "Insufficient data from CoinGecko for %s: got %d, require %d",
                    symbol, len(df) if df is not None else 0, min_required
                )
                    return None

                logger.info(
                    "Successfully fetched %d data points from CoinGecko for %s",
                    len(df), symbol
                )
                return df

    @async_rate_limiter(max_calls=120, period=60)  # yfinance scraping
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def fetch_yfinance_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetches data from yfinance asynchronously with exponential backoff."""
        logger.debug("Attempting to fetch yfinance data for %s (%s)", symbol, timeframe)

        def _sync_yf_download():
            yf_symbol = (
                f"{symbol.upper()}-USD"
                if not symbol.upper().endswith('-USD')
                else symbol.upper()
            )
            # Use the period and interval from the definitive TIMEFRAME_MAP
            config = simple_config.TIMEFRAME_MAP[timeframe]
            period = config.get('period', '365d')
            # yfinance uses '1d', '1h', etc. We need to ensure our config matches that or map it.
            # For now, assuming the 'interval' key will be added or is implicitly handled.
            # Let's add a default interval mapping for safety.
            interval_map = {'30m': '30m', '1h': '1h', '4h': '4h', '24h': '1d'}
            interval = interval_map.get(timeframe, '1h')

            data = yf.download(
                tickers=yf_symbol, period=period, interval=interval,
                progress=False, timeout=20
            )
            if data is None or data.empty:
                raise IOError("yfinance returned no data.") # Raise error to trigger tenacity retry
            return self._process_yfinance_data(data)

        processed_df = await asyncio.to_thread(_sync_yf_download)

        # Dynamically calculate minimum required samples
        min_required = simple_config.TIMEFRAME_MAP[timeframe]['lookback'] * 4
        if processed_df is None or processed_df.empty or len(processed_df) < min_required:
            logger.warning(
                "Insufficient data from yfinance for %s: got %d, require %d",
                symbol, len(processed_df) if processed_df is not None else 0, min_required
            )
            return None

        logger.info(
            "Successfully fetched %d data points from yfinance for %s",
            len(processed_df), symbol
        )
        return processed_df

    # --- Helper and Processing Functions ---

    async def get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Fetches the CoinGecko ID for a given symbol."""
        symbol_lower = symbol.lower()
        # --- Manual async-safe cache lookup ---
        cached_id = coin_id_cache.get(symbol_lower)
        if cached_id is not None:
            return cached_id
        try:
            # Include demo API key header if available to leverage higher rate limits
            headers = {"x-cg-demo-api-key": self.coingecko_api_key} if self.coingecko_api_key else None
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.coingecko.com/api/v3/search?query={symbol_lower}",
                    timeout=15,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data and 'coins' in data and data['coins']:
                        for coin in data['coins']:
                            if coin.get('symbol', '').lower() == symbol_lower:
                                coin_id = coin['id']
                            # Cache the resolved ID for 24h in TTLCache
                            coin_id_cache[symbol_lower] = coin_id
                            return coin_id
            logger.warning("Could not find CoinGecko ID for symbol: %s", symbol)
            return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error("Error fetching CoinGecko ID for %s: %s", symbol, e)
            return None

    @async_rate_limiter(max_calls=5, period=60)  # Alpha Vantage free tier
    async def fetch_alpha_vantage_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for `symbol` using the Alpha Vantage API (crypto endpoints)."""
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            logger.debug("Alpha Vantage API key not configured – skipping fetch.")
            return None

        logger.debug("Attempting Alpha Vantage fetch for %s (%s)", symbol, timeframe)

        def _sync_av_download() -> Optional[pd.DataFrame]:
            try:
                from alpha_vantage.cryptocurrencies import CryptoCurrencies
                from alpha_vantage.timeseries import TimeSeries
            except ImportError as ie:
                logger.warning("alpha_vantage package not available: %s", ie)
                return None

            sym_uc = symbol.upper()
            try:
                if timeframe == "24h":
                    cc = CryptoCurrencies(key=api_key, output_format="pandas")
                    raw_df, _ = cc.get_digital_currency_daily(symbol=sym_uc, market="USD")
                else:
                    ts = TimeSeries(key=api_key, output_format="pandas")
                    interval_map = {"30m": "30min", "1h": "60min", "4h": "60min"}
                    interval = interval_map.get(timeframe, "60min")
                    pair = f"{sym_uc}USD"
                    raw_df, _ = ts.get_intraday(symbol=pair, interval=interval, outputsize="full")
                return raw_df
            except Exception as exc:
                logger.error("Alpha Vantage synchronous fetch failed for %s: %s", symbol, exc)
                return None

        raw_df = await asyncio.to_thread(_sync_av_download)
        if raw_df is None or raw_df.empty:
            return None
        return self._process_alpha_vantage_data(raw_df, timeframe)

    def _process_alpha_vantage_data(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Standardise Alpha Vantage dataframe to OHLCV and resample."""
        # Map possible column names (daily vs intraday)
        column_map = {}
        for col in df.columns:
            lowered = col.lower()
            if "open" in lowered:
                column_map[col] = "Open"
            elif "high" in lowered:
                column_map[col] = "High"
            elif "low" in lowered:
                column_map[col] = "Low"
            elif "close" in lowered and "close" == lowered.split()[1]:  # handles daily labels
                column_map[col] = "Close"
            elif "volume" in lowered:
                column_map[col] = "Volume"
        df = df.rename(columns=column_map)
        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in required if c in df.columns]].copy()
        if df.empty or len(df.columns) < 4:
            return None
        # Index handling
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.sort_index(inplace=True)

        if timeframe in {"30m", "1h", "4h"}:
            freq_map = {"30m": "30T", "1h": "1H", "4h": "4H"}
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            df = df.resample(freq_map[timeframe]).agg(agg).dropna(how="any")

        df = df[df["Volume"] > 0]
        return df if not df.empty else None

    def _process_yfinance_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df.empty:
            return None
        df_processed = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed.index = pd.to_datetime(df_processed.index)
        if df_processed.index.tz is None:
            df_processed.index = df_processed.index.tz_localize('UTC')
        else:
            df_processed.index = df_processed.index.tz_convert('UTC')
        df_processed.dropna(inplace=True)
        df_processed = df_processed[df_processed['Volume'] > 0]
        return df_processed if not df_processed.empty else None

    def _process_coingecko_data(self, data: Dict[str, Any], timeframe: str) -> Optional[pd.DataFrame]:
        """Convert CoinGecko raw json into an OHLCV DataFrame resampled to the requested timeframe."""
        if not data.get("prices"):
            return None

        # Build initial close-price series
        df = pd.DataFrame(data["prices"], columns=["timestamp", "Close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Map our supported timeframes to pandas offset aliases
        freq_map = {"30m": "30T", "1h": "1H", "4h": "4H", "24h": "1D"}
        resample_freq = freq_map.get(timeframe, "1H")

        # Resample close into OHLC structure
        ohlc_df = df["Close"].resample(resample_freq).ohlc()

        # Handle volume if available
        if "total_volumes" in data and data["total_volumes"]:
            volume_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "Volume"])
            volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms", utc=True)
            volume_df.set_index("timestamp", inplace=True)
            volume_df = volume_df.resample(resample_freq).sum()
            final_df = ohlc_df.join(volume_df, how="inner")
        else:
            final_df = ohlc_df
            final_df["Volume"] = 0

        # Rename to standard column set
        final_df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

        # Basic cleaning
        final_df.dropna(inplace=True)
        final_df = final_df[final_df["Volume"] > 0]
        final_df.sort_index(inplace=True)

        return final_df if not final_df.empty else None

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and apply intelligent feature selection."""
        # Ensure proper column names for ta library
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Generate full suite of TA features
        df_with_ta = ta.add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )

        # ---------------- Feature Selection ---------------- #
        # Legacy best model keeps a richer indicators set; we therefore skip aggressive
        # low-variance and high-correlation pruning. If future experiments require
        # pruning, this can be toggled via a function parameter.
        df_reduced = df_with_ta.copy()

        # 3) Forward/backward fill any remaining NaNs
        df_reduced = df_reduced.ffill().bfill()

        # Ensure essential OHLCV columns are present
        essential_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in essential_cols:
            if col not in df_reduced.columns:
                df_reduced[col] = df_with_ta[col]

        return df_reduced
