import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
import time
from datetime import datetime, timedelta
import pytz
from typing import Optional, Dict, Any, List
import simple_config
from controllers.timeframe_config import get_min_samples_for_timeframe
from functools import wraps
import ta
from cachetools import TTLCache
import asyncio
import aiohttp
# Alpha Vantage disabled for crypto-focused system
# from alpha_vantage.timeseries import TimeSeries
# from alpha_vantage.cryptocurrencies import CryptoCurrencies

# Cache for CoinGecko ID lookups (TTL: 24 hours)
coin_id_cache = TTLCache(maxsize=1000, ttl=86400)

# Common coin mappings for popular cryptocurrencies (verified correct)
COMMON_COIN_MAPPING = {
    "btc": "bitcoin",
    "eth": "ethereum", 
    "bnb": "binancecoin",
    "xrp": "ripple",
    "ada": "cardano",
    "doge": "dogecoin",
    "sol": "solana",
    "matic": "matic-network",
    "avax": "avalanche-2",
    "dot": "polkadot",
    "shib": "shiba-inu",
    "near": "near",
    "link": "chainlink",
    "ltc": "litecoin",
    "bch": "bitcoin-cash",
    "uni": "uniswap",
    "atom": "cosmos",
    "xlm": "stellar",
    "algo": "algorand",
    "vet": "vechain",
    "fil": "filecoin",
    "trx": "tron",
    "etc": "ethereum-classic"
}

# Cache for coin ID lookups to speed up repeated requests
coin_id_cache = TTLCache(maxsize=1000, ttl=86400)  # 24-hour cache

# Setup logging
logger = logging.getLogger(__name__)

# Ensure this logger's messages are visible, especially if root logger is restrictive
if not logger.handlers:
    handler = logging.StreamHandler() # Log to console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent duplicate logging to root
logger.setLevel(logging.DEBUG) # Set to DEBUG to capture all relevant messages from this module

def rate_limit(max_per_minute: int = simple_config.MAX_REQUESTS_PER_MINUTE):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = {}
        min_interval = 60.0 / max_per_minute
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if func.__name__ in last_called:
                time_since_last = now - last_called[func.__name__]
                if time_since_last < min_interval:
                    time.sleep(min_interval - time_since_last)
            result = func(*args, **kwargs)
            last_called[func.__name__] = time.time()
            return result
        return wrapper
    return decorator

class CoinGeckoAPI:
    """Custom CoinGecko API client"""
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.rate_limit_remaining = 10  # Free tier limit
        self.rate_limit_reset = 0
        
        # Use free tier headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
    async def get_coin_market_chart_by_id(self, coin_id: str, vs_currency: str, days: int, desired_api_interval: str = 'daily') -> Dict:
        """Get historical market data for a coin. 
           Desired interval can be 'hourly' or 'daily'.
           CoinGecko defaults to hourly for <=90 days, daily otherwise.
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            
            params = {
                'vs_currency': vs_currency,
                'days': str(days) # CoinGecko expects days as string
            }

            # Only specify interval if we want daily AND days > 90, or if we explicitly want daily for short periods.
            # If days <= 90 and desired_api_interval is 'hourly', we omit `interval` for CoinGecko to default to hourly.
            if not (desired_api_interval == 'hourly' and days <= 90):
                params['interval'] = 'daily'
            
            logger.debug(f"CoinGecko request: coin_id={coin_id}, days={days}, effective_api_interval_param={params.get('interval')}")

            await asyncio.sleep(6)  # Free tier rate limit
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds")
                await asyncio.sleep(retry_after)
                response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 401:
                logger.warning("Using fallback simple price endpoint")
                return await self._get_fallback_data(coin_id, vs_currency, days)
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request failed: {str(e)}")
            return None
            
    async def _get_fallback_data(self, coin_id: str, vs_currency: str, days: int) -> Dict:
        """Fallback to simple price endpoint for free tier"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': vs_currency,
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data and coin_id in data:
                price = data[coin_id][vs_currency]
                market_cap = data[coin_id].get(f'{vs_currency}_market_cap', 0)
                volume = data[coin_id].get(f'{vs_currency}_24h_vol', 0)
                
                # Generate historical-like data
                now = datetime.now(pytz.UTC)
                timestamps = [(now - timedelta(days=i)).timestamp() * 1000 for i in range(min(days, 30))]
                
                return {
                    'prices': [[ts, price] for ts in timestamps],
                    'market_caps': [[ts, market_cap] for ts in timestamps],
                    'total_volumes': [[ts, volume] for ts in timestamps]
                }
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko fallback request failed: {str(e)}")
            return None

class DataFetcher:
    def __init__(self):
        self.coingecko_client = CoinGeckoAPI()
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5 minute cache
        
        # Alpha Vantage disabled for crypto-focused system
        # We only use CoinGecko and yfinance for crypto data
        self.alpha_vantage_ts_client = None
        self.alpha_vantage_crypto_client = None
        logger.info("Alpha Vantage disabled - using CoinGecko and yfinance only")
        
    async def get_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical data, first trying CoinGecko and falling back to Yahoo Finance.
        Applies technical indicators and basic cleaning.
        """
        logger.info(f"Fetching historical data for {symbol}, timeframe {timeframe}")
        try:
            df = await self.get_merged_data(symbol, timeframe)

            if df is None or df.empty:
                logger.warning(f"No data returned from get_merged_data for {symbol} ({timeframe}) an empty DataFrame will be returned.")
                return pd.DataFrame()

            min_points = self._get_min_points(timeframe)
            if len(df) < min_points:
                logger.warning(f"Insufficient data after merging for {symbol} ({timeframe}): got {len(df)}, need {min_points}. An empty DataFrame will be returned.")
                return pd.DataFrame()

            # Add technical indicators
            df_with_indicators, actual_feature_names = self._add_technical_indicators(df.copy(), timeframe)
            
            # Validate and clean final data
            final_df = self._clean_data(df_with_indicators.copy())
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error in get_historical_data: {str(e)}")
            raise

    async def _fetch_coingecko_with_retry(self, symbol: str, timeframe: str, min_points: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data from CoinGecko with retry mechanism"""
        for attempt in range(max_retries):
            try:
                coin_id = await self.get_coingecko_id(symbol.lower())
                if not coin_id:
                    logger.warning(f"No CoinGecko ID mapping for {symbol}")
                    return None
                
                # Determine desired API interval based on our system's timeframe
                # If our system timeframe is sub-daily, we desire hourly from CoinGecko if possible.
                desired_api_interval = 'hourly' if timeframe in ["30m", "1h", "4h"] else 'daily'

                # Calculate days needed. For hourly, CoinGecko provides it up to 90 days.
                # For daily, it can go longer (though our processing might cap it too).
                if desired_api_interval == 'hourly':
                    # For now, let's simplify: request up to 90 days if hourly is desired.
                    # The `min_points` check after fetching will determine sufficiency.
                    days_to_request_cg = min(self._calculate_days_needed(timeframe, min_points, source_hint='coingecko_hourly'), 90)
                else: # desired_api_interval == 'daily'
                    days_to_request_cg = self._calculate_days_needed(timeframe, min_points, source_hint='coingecko_daily')
                    # CoinGecko's free tier market_chart is often limited in total span for daily too, e.g. 365 for some, less for others
                    # Let CoinGeckoAPI handle capping days if there's an overall limit, e.g. min(days, 90) was there before for all.
                    # The days parameter to CoinGecko should be based on its capability for the desired interval.
                    # If desired is hourly, we cap days at 90. If daily, we can ask for more (e.g. up to 365).
                    # This nested if/else was causing the redundant call, simplified below.
                    # if desired_api_interval == 'hourly':
                    #     days_to_request_cg = min(self._calculate_days_needed(timeframe, min_points), 90)
                    # else: # daily
                    #     days_to_request_cg = self._calculate_days_needed(timeframe, min_points) 
                    #     days_to_request_cg = min(days_to_request_cg, 365) # Cap daily requests to 1 year for safety
                    days_to_request_cg = min(days_to_request_cg, 365) # Cap daily requests to 1 year for safety, applied to the already hint-calculated days

                logger.debug(f"CoinGecko fetch: timeframe={timeframe}, desired_api_interval={desired_api_interval}, days_to_request_cg={days_to_request_cg}, min_points_system={min_points}")

                data = await self.coingecko_client.get_coin_market_chart_by_id(
                    coin_id,
                    'usd',
                    days=days_to_request_cg, # Use calculated days
                    desired_api_interval=desired_api_interval # Pass desired interval
                )
                
                if data and 'prices' in data:
                    df = self._process_coingecko_data(data, timeframe)
                    if df is not None and len(df) >= min_points:
                        return df
                    
                logger.warning(f"Insufficient data from CoinGecko: got {len(df) if df is not None else 0} points, need {min_points}")
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"CoinGecko fetch attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
        return None

    def _calculate_days_needed(self, timeframe: str, min_points: int, source_hint: Optional[str] = None) -> int:
        """Calculate days needed based on timeframe and minimum points.
           source_hint can be used if calculation differs by source (e.g. 'coingecko_hourly')
        """
        # Default points per day for each timeframe
        # For hourly sources, we might need to adjust this if min_points refers to a specific interval
        if timeframe == "1h": points_per_day_for_timeframe = 24
        elif timeframe == "4h": points_per_day_for_timeframe = 6
        elif timeframe == "30m": points_per_day_for_timeframe = 48 # Assuming 30min resolution
        elif timeframe == "24h": points_per_day_for_timeframe = 1
        elif timeframe == "7d": points_per_day_for_timeframe = 1/7.0
        else: points_per_day_for_timeframe = 1 # Default to daily if unknown

        if points_per_day_for_timeframe == 0: calculated_days = min_points # Avoid division by zero for very coarse timeframes
        else: calculated_days = (min_points / points_per_day_for_timeframe)

        # CoinGecko free API provides max 90 days of hourly data.
        # If we want hourly data from coingecko, and min_points requires more than 90 days of hourly,
        # then coingecko cannot satisfy it with hourly data.
        # This logic should ideally be in _fetch_coingecko_with_retry to decide if it CAN fetch what's needed.
        # Here, we just calculate days based on points and interval implied by timeframe.

        # Ensure a minimum number of days, e.g., CoinGecko might need at least 1 day.
        # And the free tier was previously capped at 30 days as a minimum for demo API in old code.
        # For now, let's ensure at least a few days if calculated is too low.
        final_days = max(2, int(round(calculated_days))) # Request at least 2 days of data
        
        # The old CoinGeckoAPI specific cap `min(days, 90)` is now handled by the caller or CoinGeckoAPI itself
        # based on desired_api_interval. Max 30 used to be for demo API.
        # The `max(30, ...)` in old code was probably to ensure enough span for daily data on free tier.
        # If we are fetching hourly, we may not want to force 30 days if fewer are needed and available.
        # If we are fetching daily, 30 days might be a good minimum if min_points is low.
        if source_hint == 'coingecko_daily': # Check only source_hint
             final_days = max(30, final_days) # If daily for coingecko, ensure at least 30 days span.

        logger.debug(f"_calculate_days_needed for {timeframe}, {min_points}pts, source_hint='{source_hint}': pts_per_day={points_per_day_for_timeframe:.2f}, calc_days={calculated_days:.2f}, final_days={final_days}")
        return final_days

    async def get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Dynamically find CoinGecko ID for any crypto symbol with caching"""
        symbol = symbol.lower()
        
        # Check cache first
        if symbol in coin_id_cache:
            logger.debug(f"Found {symbol} in cache: {coin_id_cache[symbol]}")
            return coin_id_cache[symbol]
        
        # Check common mappings first (verified correct mappings)
        if symbol in COMMON_COIN_MAPPING:
            coin_id_cache[symbol] = COMMON_COIN_MAPPING[symbol]
            logger.debug(f"Found {symbol} in common mapping: {COMMON_COIN_MAPPING[symbol]}")
            return COMMON_COIN_MAPPING[symbol]
        
        # Dynamic lookup for any other coin
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/coins/list"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        coins = await response.json()
                        
                        # Look for exact symbol match
                        for coin in coins:
                            if coin['symbol'].lower() == symbol:
                                coin_id_cache[symbol] = coin['id']
                                logger.info(f"Found CoinGecko ID for {symbol}: {coin['id']}")
                                return coin['id']
                        
                        # Cache as None if not found
                        coin_id_cache[symbol] = None
                        logger.warning(f"No CoinGecko ID found for symbol: {symbol}")
                        return None
                    else:
                        logger.error(f"Failed to fetch CoinGecko coins list: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko ID for {symbol}: {e}")
            return None

    async def _fetch_yfinance_with_retry(self, symbol: str, timeframe: str, min_points: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        logger.debug(f"Attempting to fetch yfinance data for {symbol}, timeframe {timeframe}, requiring {min_points} points.")
        
        # Determine if this is a stock symbol (common stock symbols that don't need -USD)
        stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY', 'QQQ', 'IWM']
        crypto_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'SOL', 'MATIC', 'AVAX', 'DOT', 'SHIB', 'NEAR', 'LINK', 'LTC', 'BCH']
        
        # Only treat as stock if explicitly in stock_symbols list, never treat crypto as stock
        is_stock = symbol.upper() in stock_symbols and symbol.upper() not in crypto_symbols
        
        for attempt in range(max_retries):
            try:
                period = self._get_yfinance_period(timeframe, min_points)
                interval = self._get_yfinance_interval(timeframe)
                
                # Use different symbol format for stocks vs crypto
                # Avoid double -USD suffix for symbols that already contain -USD
                if symbol.upper().endswith('-USD'):
                    yf_symbol = symbol.upper()
                else:
                    yf_symbol = symbol.upper() if is_stock else f"{symbol.upper()}-USD"
                logger.debug(f"Yfinance request: symbol={yf_symbol}, period={period}, interval={interval}, attempt={attempt + 1}")
                
                ticker = yf.Ticker(yf_symbol) # Use determined symbol format
                data_hist = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=False, # Set to False to get OHLCV and Adj Close separately
                    actions=False,     # No dividends/splits needed for now
                    timeout=30 # Increased timeout
                )
                
                if not data_hist.empty:
                    logger.info(f"Fetched {len(data_hist)} raw data points from YFinance for {symbol} (attempt {attempt + 1}).")
                    logger.info(f"Data range: {data_hist.index.min()} to {data_hist.index.max()}")
                    processed_data = self._process_yfinance_data(data_hist.copy()) # Pass a copy
                    
                    if processed_data is not None and not processed_data.empty:
                        logger.info(f"Processed YFinance data for {symbol}: {len(processed_data)} points")
                        logger.info(f"Processed data range: {processed_data.index.min()} to {processed_data.index.max()}")
                        if len(processed_data) >= min_points:
                            logger.info(f"Sufficient YFinance data obtained for {symbol} ({len(processed_data)} >= {min_points}).")
                            return processed_data
                        else:
                            logger.warning(f"Insufficient YFinance data for {symbol} after processing: got {len(processed_data)}, require {min_points}. Retrying if possible.")
                    else:
                        logger.warning(f"YFinance data for {symbol} became empty or None after processing. Retrying if possible.")
                else:
                    logger.warning(f"YFinance returned empty dataframe for {symbol} with period={period}, interval={interval} (attempt {attempt + 1}).")

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt + 1) # Exponential backoff
                
            except Exception as e:
                logger.error(f"YFinance fetch attempt {attempt + 1} for {symbol} failed: {str(e)}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt + 2) # Longer backoff on error
                    
        logger.error(f"YFinance fetch definitively failed for {symbol} after {max_retries} attempts.")
        return None

    def _process_yfinance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty:
                logger.warning("YFinance: _process_yfinance_data received an empty DataFrame.")
                return pd.DataFrame()
            
            # Handle MultiIndex columns from yfinance (when downloading single ticker)
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                logger.debug(f"YFinance: Detected MultiIndex columns with {df.columns.nlevels} levels. Flattening...")
                df.columns = df.columns.droplevel(1)
            
            logger.debug(f"YFinance: Initial processing of {len(df)} rows. Columns: {df.columns.tolist()}")
            original_count = len(df)

            # Ensure DateTimeIndex and UTC
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            df.sort_index(inplace=True)

            # Standardize column names
            rename_map = {}
            for col_name in df.columns:
                col_lower = col_name.lower()
                if col_lower == 'open': rename_map[col_name] = 'Open'
                elif col_lower == 'high': rename_map[col_name] = 'High'
                elif col_lower == 'low': rename_map[col_name] = 'Low'
                elif col_lower == 'close': rename_map[col_name] = 'Close'
                elif col_lower == 'adj close': rename_map[col_name] = 'Adj Close'
                elif col_lower == 'volume': rename_map[col_name] = 'Volume'
            df.rename(columns=rename_map, inplace=True)

            # Prioritize 'Adj Close' if available
            if 'Adj Close' in df.columns and not df['Adj Close'].isnull().all():
                logger.debug("YFinance: Using 'Adj Close' as 'Close'.")
                df['Close'] = df['Adj Close']
            elif 'Close' not in df.columns or df['Close'].isnull().all():
                logger.error("YFinance: 'Close' column is missing or all NaN, cannot proceed.")
                return pd.DataFrame()

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_req_cols = [col for col in required_cols if col not in df.columns]
            if missing_req_cols:
                logger.error(f"YFinance data missing required columns after renaming: {missing_req_cols}. Has: {df.columns.tolist()}")
                return pd.DataFrame()

            # Handle NaNs in OHLCV
            df.dropna(subset=['Close'], inplace=True)
            if df.empty:
                logger.warning("YFinance: Data empty after dropping rows with NaN 'Close'.")
                return pd.DataFrame()
            
            for col in ['Open', 'High', 'Low']:
                if col in df.columns:
                    df[col] = df[col].fillna(df['Close'])
            
            df = df.ffill().bfill()
            
            df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
            df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

            zero_volume_count = (df['Volume'] <= 0).sum()
            if zero_volume_count > 0:
                logger.debug(f"YFinance: Found {zero_volume_count} rows with Volume <= 0.")
                if zero_volume_count < len(df) * 0.5:
                    df = df[df['Volume'] > 0]
                    logger.debug(f"YFinance: Removed zero volume rows. Count after: {len(df)}")
                else:
                    logger.warning(f"YFinance: More than 50% of rows have zero volume ({zero_volume_count}/{len(df)}). Not filtering by volume.")
            
            if df.empty:
                logger.warning("YFinance: Data became empty after processing (NaN drop or zero volume filter).")
                return pd.DataFrame()
            
            df = df[required_cols]

            logger.info(f"YFinance: Processed {original_count} raw rows into {len(df)} cleaned rows.")
            return df
        except Exception as e_process:
            logger.error(f"YFinance: Unhandled exception in _process_yfinance_data: {str(e_process)}", exc_info=True)
            return pd.DataFrame()

    def _process_coingecko_data(self, data: Dict[str, Any], timeframe: str) -> Optional[pd.DataFrame]:
        """Process CoinGecko data to standardized OHLCV format."""
        try:
            if not data or not isinstance(data, dict):
                logger.error("CoinGecko: Invalid data format received")
                return None
            
            # Extract price and volume data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                logger.error("CoinGecko: No price data in response")
                return None
            
            # Create DataFrame with timestamps
            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Add volume data if available
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms', utc=True)
                volume_df.set_index('timestamp', inplace=True)
                df['Volume'] = volume_df['Volume']
            else:
                df['Volume'] = 0
            
            # Generate OHLC from close prices
            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close'].rolling(2, min_periods=1).max()
            df['Low'] = df['Close'].rolling(2, min_periods=1).min()
            
            # Handle initial NaNs from shifts or missing volume
            df = df.ffill().bfill()

            # Ensure required columns exist after processing
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"CoinGecko: Processed data missing required columns: {missing_cols}. Has: {df.columns.tolist()}")
                # Attempt to fill missing essential OHLC with Close, Volume with 0
                for col in missing_cols:
                    if col in ['Open', 'High', 'Low']:
                        df[col] = df['Close']
                    elif col == 'Volume':
                        df[col] = 0
                # Recheck
                if not all(col in df.columns for col in required_cols):
                     logger.error(f"CoinGecko: Still missing columns after fallback fill: {[c for c in required_cols if c not in df.columns]}. Returning None.")
                     return None
            
            df = df[required_cols]

            # Remove rows where essential data might still be NaN
            df.dropna(subset=['Close', 'Volume'], inplace=True)
            df = df[df['Volume'] >= 0]
            
            if df.empty:
                logger.warning("CoinGecko: Data empty after processing")
                return None
            
            logger.info(f"Successfully processed CoinGecko data. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing CoinGecko data: {str(e)}")
            return None

    def _add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> tuple:
        """Add enhanced technical indicators to DataFrame with correlation handling"""
        try:
            # Import enhanced feature processor
            from controllers.enhanced_feature_processor import EnhancedFeatureProcessor, get_enhanced_feature_list
            
            # Create enhanced feature processor
            processor = EnhancedFeatureProcessor()
            
            # Create a copy for technical indicators
            df_ta = df.copy()
            
            # Add traditional indicators first
            df_ta['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df_ta['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df_ta['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
            
            # Add MACD
            macd = ta.trend.MACD(df['Close'])
            df_ta['MACD'] = macd.macd()
            df_ta['MACD_Signal'] = macd.macd_signal()
            df_ta['MACD_Hist'] = macd.macd_diff()
            
            # Add Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df_ta['Bollinger_middle'] = bollinger.bollinger_mavg()
            df_ta['Bollinger_Upper'] = bollinger.bollinger_hband()
            df_ta['Bollinger_Lower'] = bollinger.bollinger_lband()
            
            # Add ATR and other volume indicators
            df_ta['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            df_ta['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df_ta['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            df_ta['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            # Add Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df_ta['Stoch_%K'] = stoch.stoch()
            df_ta['Stoch_%D'] = stoch.stoch_signal()
            
            # Add MFI
            df_ta['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], 
                                                    window=min(14, len(df)-1))
            
            # Add enhanced technical indicators
            df_enhanced = processor.add_enhanced_technical_indicators(df_ta)
            
            # Handle missing values
            df_enhanced = df_enhanced.ffill().bfill()
            
            # Clean and handle correlations
            df_cleaned, scaler = processor.clean_and_scale_features(df_enhanced, target_column='Close')
            
            # Get final feature list (excluding target)
            feature_columns = [col for col in df_cleaned.columns if col != 'Close']
            feature_columns = ['Close'] + feature_columns  # Put Close first for consistency
            
            # Ensure all features are available
            available_features = [col for col in feature_columns if col in df_cleaned.columns]
            
            logger.info(f"Generated {len(available_features)} enhanced technical indicators for {timeframe} timeframe")
            logger.info(f"Removed {len(processor.removed_features)} correlated/low-variance features")
            
            return df_cleaned[available_features], available_features
            
        except Exception as e:
            logger.error(f"Error adding enhanced technical indicators: {str(e)}")
            # Fallback to original implementation
            return self._add_technical_indicators_fallback(df, timeframe)
    
    def _add_technical_indicators_fallback(self, df: pd.DataFrame, timeframe: str) -> tuple:
        """Fallback technical indicators implementation"""
        try:
            # Ensure we have the basic OHLCV columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required OHLCV columns: {missing_columns}")
                return None, None
            
            # Create a copy for technical indicators
            df_ta = df.copy()
            
            # Add basic indicators
            df_ta['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df_ta['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df_ta['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
            
            # Add MACD
            macd = ta.trend.MACD(df['Close'])
            df_ta['MACD'] = macd.macd()
            df_ta['MACD_Signal'] = macd.macd_signal()
            df_ta['MACD_Hist'] = macd.macd_diff()
            
            # Add ATR (only if High/Low are available)
            if 'High' in df.columns and 'Low' in df.columns:
                df_ta['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            else:
                df_ta['ATR'] = df['Close'].rolling(window=14).std()  # Fallback volatility measure
            
            # Add OBV
            df_ta['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            # Handle missing values in technical indicators
            df_ta = df_ta.ffill().bfill()
            
            # Get the list of feature columns in the correct order
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'RSI_14', 
                             'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'OBV']
            
            # Filter to available columns
            available_features = [col for col in feature_columns if col in df_ta.columns]
            
            logger.info(f"Generated {len(available_features)} fallback technical indicators for {timeframe} timeframe")
            return df_ta[available_features], available_features
            
        except Exception as e:
            logger.error(f"Error in fallback technical indicators: {str(e)}")
            return None, None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        try:
            # Remove rows with zero volume
            df = df[df['Volume'] > 0].copy()
            
            # Forward fill missing values
            df = df.ffill().bfill()
            
            # Remove duplicate indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    async def get_coingecko_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from CoinGecko with improved error handling and rate limiting"""
        try:
            coin_id = await self.get_coingecko_id(symbol)
            if not coin_id:
                logger.warning(f"No CoinGecko ID mapping for {symbol}")
                return None

            # Get period from TIMEFRAME_MAP, ensure it doesn't exceed 365 days
            period_str = simple_config.TIMEFRAME_MAP[timeframe].get('period', '30d')
            days = int(period_str[:-1])
            if days > 365:
                logger.warning(f"Adjusting period from {days} to 365 days due to CoinGecko free tier limitation")
                days = 365

            url = f"{simple_config.COINGECKO_API_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': str(days)
                # Removed interval parameter as it's restricted in demo API
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds before retry...")
                        await asyncio.sleep(retry_after)
                        async with session.get(url, params=params, timeout=30) as retry_response:
                            response = retry_response

                    if response.status == 200:
                        data = await response.json()
                        if not data or 'prices' not in data:
                            logger.error(f"Invalid response format from CoinGecko for {symbol}")
                            return None

                        df = self._process_coingecko_data(data, timeframe)
                        if df is None:
                            logger.warning(f"Failed to process CoinGecko data for {symbol}")
                            return None
                        
                        min_required = get_min_samples_for_timeframe(timeframe)
                        if len(df) >= min_required:
                            logger.info(f"Successfully fetched and processed CoinGecko data for {symbol}, {len(df)} points.")
                            return df
                        else:
                            logger.warning(f"Insufficient data points from CoinGecko for {symbol}: got {len(df)}, require {min_required}")
                            return None

                    elif response.status == 401:
                        error_text = await response.text()
                        logger.error(f"CoinGecko API authentication failed: {error_text}")
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"CoinGecko API request failed with status {response.status}: {error_text}")
                        return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching CoinGecko data for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching CoinGecko data for {symbol}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_coingecko_data for {symbol}: {str(e)}")
            return None

    def fetch_yfinance_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance with improved error handling"""
        try:
            # Determine if this is a stock symbol (common stock symbols that don't need -USD)
            stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY', 'QQQ', 'IWM']
            # Only treat as stock if explicitly in the stock_symbols list - don't use generic length rules
            is_stock = symbol.upper() in stock_symbols
            
            # Avoid double -USD suffix for symbols that already contain -USD
            if symbol.upper().endswith('-USD'):
                yf_symbol = symbol.upper()
            else:
                yf_symbol = symbol.upper() if is_stock else f"{symbol}-USD"
            
            # Use period and interval from TIMEFRAME_MAP
            period_str = simple_config.TIMEFRAME_MAP[timeframe].get('period', '365d')
            interval = simple_config.TIMEFRAME_MAP[timeframe].get('interval', '1h')
            
            logger.debug(f"Requesting {period_str} of data for {symbol} from yfinance with {interval} interval.")
            
            # Convert period string to number of days
            try:
                num_days = int(period_str[:-1])
                if num_days > 365 and interval in ['30m', '1h', '4h']:
                    # yfinance has limitations for intraday data
                    num_days = 365
                    logger.warning(f"Adjusted request period to 365 days due to yfinance limitations for {interval} data")
            except ValueError:
                logger.error(f"Invalid period format in TIMEFRAME_MAP for {timeframe}: {period_str}")
                num_days = 365
            
            end = datetime.now(pytz.UTC)
            start = end - timedelta(days=num_days)
            
            # Download data
            data = yf.download(
                tickers=yf_symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False
            )
            
            if data is None or data.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return None
            
            logger.info(f"Successfully fetched yfinance data for {symbol}, {len(data)} points before processing.")
            
            # Process data
            processed_df = self._process_yfinance_data(data)
            logger.debug(f"Processed yfinance data, {len(processed_df)} rows")
            
            # Verify minimum samples
            min_required = get_min_samples_for_timeframe(timeframe)
            if len(processed_df) < min_required:
                logger.warning(f"Insufficient data points from yfinance for {symbol}: got {len(processed_df)}, require {min_required}")
                return None
            
            logger.info(f"Successfully processed yfinance data for {symbol}, {len(processed_df)} points.")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {str(e)}")
            return None

    async def get_merged_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetches data by trying CoinGecko, then yfinance, then Alpha Vantage.
        """
        cache_key = f"{symbol}_{timeframe}_merged"
        cached_df = self.cache.get(cache_key)
        if cached_df is not None:
            logger.info(f"Returning cached merged data for {symbol} ({timeframe})")
            return cached_df.copy()

        min_points = self._get_min_points(timeframe)
        df = None
        source_used = None

        # 1. Try CoinGecko
        logger.info(f"Attempting to fetch data from CoinGecko for {symbol} ({timeframe})")
        df_coingecko = await self._fetch_coingecko_with_retry(symbol, timeframe, min_points)
        if df_coingecko is not None and not df_coingecko.empty and len(df_coingecko) >= min_points:
            df = df_coingecko
            source_used = "CoinGecko"
            logger.info(f"Using data from CoinGecko for {symbol} ({timeframe}). Points: {len(df)}")
        else:
            logger.warning(f"CoinGecko failed or returned insufficient data for {symbol} ({timeframe}). Trying yfinance.")

            # 2. Try yfinance if CoinGecko fails
            df_yfinance = await self._fetch_yfinance_with_retry(symbol, timeframe, min_points)
            if df_yfinance is not None and not df_yfinance.empty and len(df_yfinance) >= min_points:
                df = df_yfinance
                source_used = "yfinance"
                logger.info(f"Using data from yfinance for {symbol} ({timeframe}). Points: {len(df)}")
            else:
                logger.warning(f"yfinance also failed or returned insufficient data for {symbol} ({timeframe}).")
                
                # 3. Try Alpha Vantage ONLY for stocks (not crypto) and if client is available
                if self._is_stock_symbol(symbol) and self.alpha_vantage_ts_client is not None:
                    logger.info(f"Trying Alpha Vantage for stock symbol {symbol}")
                    df_alphavantage = await self._fetch_alphavantage_with_retry(symbol, timeframe, min_points)
                    if df_alphavantage is not None and not df_alphavantage.empty and len(df_alphavantage) >= min_points:
                        df = df_alphavantage
                        source_used = "AlphaVantage"
                        logger.info(f"Using data from Alpha Vantage for {symbol} ({timeframe}). Points: {len(df)}")
                    else:
                        logger.warning(f"Alpha Vantage also failed or returned insufficient data for stock {symbol} ({timeframe}).")
                else:
                    logger.info(f"Skipping Alpha Vantage for crypto symbol {symbol} - using CoinGecko/yfinance only for crypto")

        if df is not None and not df.empty:
            # Add technical indicators AFTER a successful fetch from any source
            df_with_indicators, actual_feature_names = self._add_technical_indicators(df.copy(), timeframe)
            self.cache[cache_key] = df_with_indicators.copy() # Cache the processed data
            logger.info(f"Data fetched and processed for {symbol} ({timeframe}) using {source_used}. Shape: {df_with_indicators.shape}")
            return df_with_indicators
        
        logger.error(f"Failed to fetch sufficient data for {symbol} ({timeframe}) from all sources.")
        return None # Explicitly return None if all sources fail

    def _get_min_points(self, timeframe: str) -> int:
        """Get minimum required data points for a timeframe - Dynamic rule: lookback * 4"""
        if timeframe not in simple_config.TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        lookback = simple_config.TIMEFRAME_MAP[timeframe]['lookback']
        min_required = lookback * 4  # Dynamic rule for sufficient training data
        
        logger.debug(f"Min points for {timeframe}: {min_required} (lookback={lookback} * 4)")
        return min_required
        
    def _get_coingecko_interval(self, timeframe: str) -> Optional[str]:
        """Get CoinGecko interval for timeframe"""
        interval_map = {
            "1h": "hourly",
            "4h": "hourly",
            "24h": "daily",
            "7d": "daily"
        }
        return interval_map.get(timeframe)
        
    def _get_yfinance_interval(self, timeframe: str) -> str:
        """Get yfinance interval for timeframe"""
        interval_map = {
            "30m": "30m",  # Added 30m support
            "1h": "1h",
            "4h": "4h",
            "24h": "1d",
            "7d": "1d"
        }
        return interval_map.get(timeframe, "1d")
        
    def _get_yfinance_period(self, timeframe: str, min_points: int) -> str:
        """Get yfinance period based on timeframe and minimum points"""
        # Calculate minimum days needed
        days_map = {
            "30m": min_points / 48,  # 48 points per day for 30m
            "1h": min_points / 24,   # 24 points per day
            "4h": min_points / 6,    # 6 points per day
            "24h": min_points,       # 1 point per day
            "7d": min_points * 7     # 1 point per week
        }
        
        days_needed = int(days_map.get(timeframe, min_points))
        
        # YFinance limitations for intraday data
        if timeframe in ["30m", "1h", "4h"]:
            if timeframe == "30m":
                # For 30m, request up to 60 days of data
                # YFinance typically provides 60 days of 30m data reliably
                days_needed = min(max(days_needed, 45), 60)  # At least 45 days, max 60
                logger.info(f"Requesting {days_needed} days of 30m data from YFinance (expecting ~{days_needed * 48} points)")
            else:
                # For 1h and 4h, can request more
                days_needed = min(days_needed, 90)
            logger.info(f"Adjusted request period to {days_needed} days due to YFinance intraday limitations for {timeframe}")
        
        # Add a small buffer to the period
        return f"{days_needed + 3}d"

    def _get_alphavantage_symbol(self, symbol: str) -> str:
        """Formats symbol for Alpha Vantage (e.g., BTC -> BTCUSD for stocks/generic, or just BTC for crypto client)."""
        # For TimeSeries client (stocks), format might be just the symbol, e.g. MSFT
        # For CryptoCurrencies client, symbol is like 'BTC', market is 'USD'
        return symbol.upper() # Crypto client takes 'BTC', not 'BTCUSD'

    def _is_stock_symbol(self, symbol: str) -> bool:
        """Detect if symbol is a stock (not a cryptocurrency)"""
        stock_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC',
            'NFLX', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'IBM', 'UBER', 'LYFT', 'SNAP', 'TWTR',
            'FB', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI', 'DIDI', 'GME', 'AMC',
            'SPY', 'QQQ', 'VOO', 'VTI', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV'
        }
        return symbol.upper() in stock_symbols

    async def _fetch_alphavantage_with_retry(self, symbol: str, timeframe: str, min_points: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        # Check if Alpha Vantage clients are available
        if self.alpha_vantage_ts_client is None or self.alpha_vantage_crypto_client is None:
            logger.warning("Alpha Vantage API key not configured, skipping Alpha Vantage fetch")
            return None
            
        logger.debug(f"Attempting to fetch Alpha Vantage data for {symbol}, timeframe {timeframe}, requiring {min_points} points.")
        av_symbol = symbol.upper()
        data = None
        meta_data = None
        is_intraday_fetch = timeframe in ["30m", "1h", "4h"]
        is_stock = self._is_stock_symbol(symbol)

        for attempt in range(max_retries):
            try:
                if is_stock:
                    # Use TimeSeries client for stocks
                    if is_intraday_fetch:
                        logger.debug(f"Alpha Vantage Stock INTRADAY attempt {attempt+1} for {av_symbol} (target timeframe: {timeframe})")
                        # Map our timeframes to Alpha Vantage intervals
                        av_interval = {'30m': '30min', '1h': '60min', '4h': '60min'}.get(timeframe, '60min')
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_ts_client.get_intraday,
                            symbol=av_symbol,
                            interval=av_interval,
                            outputsize='full'
                        )
                    else: # Daily fetch
                        logger.debug(f"Alpha Vantage Stock DAILY attempt {attempt+1} for {av_symbol}")
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_ts_client.get_daily,
                            symbol=av_symbol,
                            outputsize='full'
                        )
                else:
                    # Use crypto client for cryptocurrencies
                    if is_intraday_fetch:
                        logger.debug(f"Alpha Vantage Crypto INTRADAY attempt {attempt+1} for {av_symbol}/USD (target timeframe: {timeframe})")
                        # The interval for intraday is not settable via this library, AV returns a fixed one (e.g. 1min, 5min)
                        # We will resample it later in _process_alphavantage_data
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_crypto_client.get_digital_currency_intraday,
                            symbol=av_symbol,
                            market='USD'
                        )
                    else: # Daily fetch
                        logger.debug(f"Alpha Vantage Crypto DAILY attempt {attempt+1} for {av_symbol}/USD")
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_crypto_client.get_digital_currency_daily,
                            symbol=av_symbol,
                            market='USD'
                        )

                if data is not None and not data.empty:
                    # Pass timeframe for potential resampling if intraday
                    if is_stock:
                        df = self._process_alphavantage_data(data, av_symbol, 
                                                             target_timeframe=timeframe,
                                                             is_crypto_daily_format=False, 
                                                             is_intraday_source=is_intraday_fetch,
                                                             is_stock=True)
                    else:
                        df = self._process_alphavantage_data(data, av_symbol, 
                                                             target_timeframe=timeframe,
                                                             is_crypto_daily_format=not is_intraday_fetch, 
                                                             is_intraday_source=is_intraday_fetch,
                                                             is_stock=False)
                    
                    if df is not None and len(df) >= min_points:
                        asset_type = "Stock" if is_stock else "Crypto"
                        fetch_type = "INTRADAY" if is_intraday_fetch else "DAILY"
                        logger.info(f"Successfully fetched {fetch_type} {asset_type.lower()} data from Alpha Vantage for {av_symbol} on attempt {attempt + 1}. Points: {len(df)}")
                        return df
                    else:
                        logger.warning(f"Insufficient data from Alpha Vantage for {av_symbol} after processing (target: {timeframe}): got {len(df) if df is not None else 0}, need {min_points}")
                else:
                    logger.warning(f"No data returned from Alpha Vantage for {av_symbol} (target: {timeframe}) on attempt {attempt + 1}.")

                if attempt < max_retries - 1:
                    await asyncio.sleep( (2 ** attempt) * 15) # Increased sleep for AV free tier (5 calls/min, 100/day)
                
            except Exception as e:
                logger.error(f"Alpha Vantage fetch attempt {attempt + 1} for {av_symbol} (target: {timeframe}) failed: {type(e).__name__} - {str(e)}", exc_info=True) # Log stack trace
                if "api key" in str(e).lower() or "invalid API call" in str(e).lower() or "higher_frequency_data" in str(e).lower(): # "higher_frequency_data" can mean sub for intraday
                    logger.error("Alpha Vantage API key issue, invalid call, or premium endpoint hit. Breaking retry loop.")
                    break 
                if attempt < max_retries - 1:
                    await asyncio.sleep( (2 ** attempt) * 20) # Longer sleep on error
            
        logger.warning(f"Failed to fetch sufficient data from Alpha Vantage for {av_symbol} (target: {timeframe}) after {max_retries} attempts.")
        return None

    def _process_alphavantage_data(self, df: pd.DataFrame, symbol_orig: str, target_timeframe: str, 
                                   is_crypto_daily_format: bool = False, 
                                   is_intraday_source: bool = False,
                                   is_stock: bool = False) -> Optional[pd.DataFrame]:
        """Process Alpha Vantage data to match expected format.
           Handles stock format, crypto daily format, and crypto intraday format.
           Resamples intraday data to the target_timeframe.
        """
        try:
            logger.debug(f"Processing Alpha Vantage data for {symbol_orig}. Target timeframe: {target_timeframe}, DailyCryptoFmt: {is_crypto_daily_format}, IntradaySrc: {is_intraday_source}, IsStock: {is_stock}. Initial Cols: {df.columns.tolist()}")
            
            # AV pandas output often has 'date' as index name for daily, or 'index' after reset.
            # Intraday get_digital_currency_intraday directly returns DataFrame with DatetimeIndex.
            if not isinstance(df.index, pd.DatetimeIndex):
                # If 'date' is a column, set it as index. Otherwise, try resetting.
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                else: # This case might happen if index is not named 'date' and not DatetimeIndex
                    df.reset_index(inplace=True) # Try to make 'date' or 'index' a column
                    date_col_found = False
                    for col_candidate in ['date', 'index', 'timestamp']: # common names for date column
                        if col_candidate in df.columns:
                            df.rename(columns={col_candidate: 'Timestamp'}, inplace=True)
                            df.set_index('Timestamp', inplace=True)
                            date_col_found = True
                            break
                    if not date_col_found:
                        logger.error(f"AV: Could not identify a date/timestamp column to set as index. Columns: {df.columns.tolist()}")
                        return None
            
            # Ensure index is DatetimeIndex and UTC
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC') # Assume UTC if not specified by AV
            else:
                df.index = df.index.tz_convert('UTC')
            df.sort_index(inplace=True)

            col_map = {}
            if is_stock:
                # Stock data columns: '1. open', '2. high', '3. low', '4. close', '5. volume'
                for col in df.columns:
                    col_lower = col.lower()
                    if '1. open' in col_lower: col_map[col] = 'Open'
                    elif '2. high' in col_lower: col_map[col] = 'High'
                    elif '3. low' in col_lower: col_map[col] = 'Low'
                    elif '4. close' in col_lower: col_map[col] = 'Close'
                    elif '5. volume' in col_lower: col_map[col] = 'Volume'
                    elif '6. adjusted close' in col_lower: col_map[col] = 'Adj Close'
                df.rename(columns=col_map, inplace=True)
                
            elif is_intraday_source:
                # Intraday crypto columns from alpha_vantage lib are: '1. open', '2. high', '3. low', '4. close', '5. volume'
                # This is for client.get_digital_currency_intraday
                for col in df.columns:
                    if '1. open' in col: col_map[col] = 'Open'
                    elif '2. high' in col: col_map[col] = 'High'
                    elif '3. low' in col: col_map[col] = 'Low'
                    elif '4. close' in col: col_map[col] = 'Close'
                    elif '5. volume' in col: col_map[col] = 'Volume'
                df.rename(columns=col_map, inplace=True)

            elif is_crypto_daily_format:
                # Intraday crypto columns from alpha_vantage lib are: '1. open', '2. high', '3. low', '4. close', '5. volume'
                # This is for client.get_digital_currency_intraday
                for col in df.columns:
                    if '1. open' in col: col_map[col] = 'Open'
                    elif '2. high' in col: col_map[col] = 'High'
                    elif '3. low' in col: col_map[col] = 'Low'
                    elif '4. close' in col: col_map[col] = 'Close'
                    elif '5. volume' in col: col_map[col] = 'Volume'
                df.rename(columns=col_map, inplace=True)

            elif is_crypto_daily_format:
                # Daily Crypto columns: '1a. open (USD)', '2a. high (USD)', etc.
                # Market symbol (e.g. USD) might be needed if not always USD
                market_suffix = "(USD)" # Assuming USD, adjust if market can vary and is in col name
                for col in df.columns:
                    col_lower = col.lower()
                    if f'1a. open {market_suffix.lower()}' in col_lower or (f'1. open {market_suffix.lower()}' in col_lower and "1a." not in col_lower) : col_map[col] = 'Open' # AV sometimes uses 1. open (USD) for daily too
                    elif f'2a. high {market_suffix.lower()}' in col_lower or (f'2. high {market_suffix.lower()}' in col_lower and "2a." not in col_lower): col_map[col] = 'High'
                    elif f'3a. low {market_suffix.lower()}' in col_lower or (f'3. low {market_suffix.lower()}' in col_lower and "3a." not in col_lower): col_map[col] = 'Low'
                    elif f'4a. close {market_suffix.lower()}' in col_lower or (f'4. close {market_suffix.lower()}' in col_lower and "4a." not in col_lower): col_map[col] = 'Close'
                    elif '5. volume' == col_lower and not market_suffix.lower() in col_lower : col_map[col] = 'Volume' # '5. volume'
                    elif f'6. market cap {market_suffix.lower()}' in col_lower : pass # Ignoring market cap for now
                df.rename(columns=col_map, inplace=True)

            else: # Original stock format (e.g., for TimeSeries client if ever used for other assets)
                df.rename(columns={
                    '1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
                    '4. close': 'Close', '5. adjusted close': 'Adj Close', 
                    '6. volume': 'Volume'
                }, inplace=True)
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']

            # Ensure all essential columns are numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else: # If a required column is missing after mapping, this is an issue
                    logger.error(f"AV: Essential column '{col}' missing after renaming. Available: {df.columns.tolist()}")
                    return None
            
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True) # Drop rows where essential values are NaN
            if df.empty:
                logger.warning(f"AV: Data for {symbol_orig} empty after mapping and NaN drop.")
                return None

            # Resample if intraday source data needs to be converted to a coarser target_timeframe
            # AV intraday is often 1-min or 5-min. Our target_timeframe could be "30m", "1h", "4h".
            if is_intraday_source and target_timeframe in ["30m", "1h", "4h"]:
                logger.debug(f"AV: Resampling intraday data for {symbol_orig} to {target_timeframe}. Original points: {len(df)}")
                
                # Map our timeframe string to pandas offset string
                resample_map = {"30m": "30T", "1h": "1H", "4h": "4H"}
                pandas_timeframe = resample_map.get(target_timeframe)

                if pandas_timeframe:
                    ohlc_dict = {
                        'Open': 'first', 'High': 'max', 'Low': 'min', 
                        'Close': 'last', 'Volume': 'sum'
                    }
                    # Only resample if there are columns to aggregate
                    cols_to_agg = {k: v for k,v in ohlc_dict.items() if k in df.columns}
                    if not cols_to_agg:
                        logger.error(f"AV: No columns to aggregate for resampling {symbol_orig} to {target_timeframe}. Columns: {df.columns.tolist()}")
                        return None

                    df = df.resample(pandas_timeframe).agg(cols_to_agg)
                    df.dropna(how='all', inplace=True) # Remove intervals that are all NaN after resampling
                    # Forward fill is important after resampling to handle periods with no trades
                    # but we also need to be careful not to fill too much into the future if data ends.
                    # df.ffill(inplace=True) # Let's see if this is needed after .agg
                    if df.empty:
                        logger.warning(f"AV: Data for {symbol_orig} empty after resampling to {target_timeframe}.")
                        return None
                    logger.debug(f"AV: Resampled to {len(df)} points for {target_timeframe}.")
                else:
                    logger.warning(f"AV: Could not map target_timeframe '{target_timeframe}' to pandas resample string. No resampling performed.")
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_final_cols = [col for col in required_cols if col not in df.columns]
            if missing_final_cols:
                logger.error(f"AV: Missing expected columns after all processing for {symbol_orig} (target {target_timeframe}): {missing_final_cols}. Available: {df.columns.tolist()}")
                return None

            # Final selection and order
            processed_df = df[required_cols].copy()
            
            # Drop rows with zero volume as a final cleaning step, if not already done
            processed_df = processed_df[processed_df['Volume'] > 0]
            if processed_df.empty:
                logger.warning(f"AV: Data for {symbol_orig} became empty after zero volume filter (target {target_timeframe}).")
                return None

            logger.info(f"Successfully processed Alpha Vantage data for {symbol_orig} (target {target_timeframe}). Shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data for {symbol_orig} (target {target_timeframe}): {str(e)}", exc_info=True)
            return None 