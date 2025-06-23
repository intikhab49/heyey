from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from controllers.data_fetcher import DataFetcher
from controllers.error_handler import error_tracker, ErrorSeverity, ErrorCategory
from controllers.data_validator import DataValidator
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj

@router.get("/info/{symbol}")
async def get_crypto_info(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 30m, 1h, 4h, 24h")
) -> Dict[str, Any]:
    """
    Get comprehensive information about a cryptocurrency
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC, ETH, DOGE)
        timeframe: Time interval for latest data
        
    Returns:
        Comprehensive crypto information including latest price, market data, and availability
    """
    try:
        symbol = symbol.upper()
        if timeframe not in ["30m", "1h", "4h", "24h"]:
            raise ValueError(f"Invalid timeframe: {timeframe}. Allowed: 30m, 1h, 4h, 24h")
        
        data_fetcher = DataFetcher()
        
        # Get basic info and latest price
        latest_data = await data_fetcher.get_data(symbol, timeframe, days=1)
        
        if latest_data is None or latest_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data available for {symbol}. Symbol may not exist or be supported."
            )
        
        # Get latest price info
        latest_row = latest_data.iloc[-1]
        
        # Calculate 24h changes
        day_ago_data = await data_fetcher.get_data(symbol, "24h", days=2)
        price_24h_ago = None
        change_24h = None
        change_24h_pct = None
        
        if day_ago_data is not None and len(day_ago_data) >= 2:
            price_24h_ago = float(day_ago_data.iloc[-2]['Close'])
            current_price = float(latest_row['Close'])
            change_24h = current_price - price_24h_ago
            change_24h_pct = (change_24h / price_24h_ago) * 100
        
        # Get data availability info
        total_data = await data_fetcher.get_data(symbol, timeframe, days=365)
        data_points = len(total_data) if total_data is not None else 0
        
        # Calculate volatility (30-day)
        volatility_data = await data_fetcher.get_data(symbol, "24h", days=30)
        volatility = None
        if volatility_data is not None and len(volatility_data) > 1:
            returns = volatility_data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(365) * 100)  # Annualized volatility %
        
        result = {
            "symbol": symbol,
            "name": symbol,  # Could be enhanced with full name lookup
            "current_price": float(latest_row['Close']),
            "price_24h_ago": price_24h_ago,
            "change_24h": change_24h,
            "change_24h_percent": change_24h_pct,
            "volume_24h": float(latest_row['Volume']),
            "high_24h": float(latest_row['High']),
            "low_24h": float(latest_row['Low']),
            "volatility_30d_percent": volatility,
            "last_updated": latest_row.name.isoformat() if hasattr(latest_row.name, 'isoformat') else str(latest_row.name),
            "data_availability": {
                "timeframe": timeframe,
                "total_data_points": data_points,
                "oldest_data": total_data.index[0].isoformat() if data_points > 0 else None,
                "latest_data": total_data.index[-1].isoformat() if data_points > 0 else None
            },
            "supported_timeframes": ["30m", "1h", "4h", "24h"],
            "status": "success"
        }
        
        return convert_numpy_types(result)
        
    except ValueError as e:
        logger.error(f"Validation error for {symbol} info: {e}")
        error_tracker.log_error(str(e), ErrorSeverity.LOW, ErrorCategory.VALIDATION)
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Error getting info for {symbol}: {e}")
        error_tracker.log_error(str(e), ErrorSeverity.MEDIUM, ErrorCategory.DATA_FETCH)
        raise HTTPException(
            status_code=500, 
            detail=f"Unable to fetch information for {symbol}: {str(e)}"
        )

@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 30m, 1h, 4h, 24h"),
    days: int = Query(30, description="Number of days of historical data", ge=1, le=365),
    format: str = Query("json", description="Response format: json or csv")
) -> Dict[str, Any]:
    """
    Get historical price data for a cryptocurrency
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC, ETH, DOGE)
        timeframe: Time interval
        days: Number of days of historical data (1-365)
        format: Response format (json or csv)
        
    Returns:
        Historical price data with OHLCV information
    """
    try:
        symbol = symbol.upper()
        if timeframe not in ["30m", "1h", "4h", "24h"]:
            raise ValueError(f"Invalid timeframe: {timeframe}. Allowed: 30m, 1h, 4h, 24h")
        
        if format not in ["json", "csv"]:
            raise ValueError(f"Invalid format: {format}. Allowed: json, csv")
        
        data_fetcher = DataFetcher()
        
        # Get historical data
        data = await data_fetcher.get_data(symbol, timeframe, days=days)
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No historical data available for {symbol}"
            )
        
        # Prepare response
        if format == "csv":
            # For CSV format, return the data as CSV string
            csv_data = data.to_csv()
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "days": days,
                "format": "csv",
                "data": csv_data,
                "total_points": len(data),
                "date_range": {
                    "start": data.index[0].isoformat(),
                    "end": data.index[-1].isoformat()
                }
            }
        else:
            # JSON format
            # Convert data to records for JSON response
            records = []
            for timestamp, row in data.iterrows():
                records.append({
                    "timestamp": timestamp.isoformat(),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": float(row['Volume'])
                })
            
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "days": days,
                "format": "json",
                "total_points": len(data),
                "date_range": {
                    "start": data.index[0].isoformat(),
                    "end": data.index[-1].isoformat()
                },
                "data": records,
                "status": "success"
            }
            
            return convert_numpy_types(result)
        
    except ValueError as e:
        logger.error(f"Validation error for {symbol} historical data: {e}")
        error_tracker.log_error(str(e), ErrorSeverity.LOW, ErrorCategory.VALIDATION)
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        error_tracker.log_error(str(e), ErrorSeverity.MEDIUM, ErrorCategory.DATA_FETCH)
        raise HTTPException(
            status_code=500, 
            detail=f"Unable to fetch historical data for {symbol}: {str(e)}"
        )

@router.get("/health")
async def data_health_check():
    """Health check for data endpoints"""
    return {
        "status": "healthy",
        "service": "crypto-data-api",
        "endpoints": [
            "/info/{symbol}",
            "/historical/{symbol}",
            "/health"
        ]
    }
        
        if df is None or df.empty:
            raise ValueError(f"No historical data found for {symbol}")
        
        # Limit the results
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        # Convert to list of dictionaries
        historical_data = []
        for index, row in df.iterrows():
            historical_data.append({
                "timestamp": index.isoformat(),
                "open": float(row.get('Open', 0)),
                "high": float(row.get('High', 0)),
                "low": float(row.get('Low', 0)),
                "close": float(row.get('Close', 0)),
                "volume": float(row.get('Volume', 0))
            })
        
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_points": len(historical_data),
            "data": historical_data
        }
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error for {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")

@router.get("/realtime/{symbol}")
async def get_realtime_price(symbol: str) -> Dict[str, Any]:
    """Get real-time price for a symbol"""
    try:
        symbol = symbol.upper()
        
        # Get current price
        current_price = await fetch_coingecko_price(symbol)
        
        if current_price is None:
            raise ValueError(f"No real-time price found for {symbol}")
        
        result = {
            "symbol": symbol,
            "price": current_price,
            "timestamp": "2025-06-20T16:30:00Z",  # Current time would be better
            "source": "coingecko"
        }
        
        return result
        
    except ValueError as e:
        logger.error(f"Price lookup error for {symbol}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting realtime price for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get realtime price: {str(e)}")
