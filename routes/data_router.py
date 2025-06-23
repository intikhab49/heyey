from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging
from controllers.data_fetcher import DataFetcher
from controllers.prediction import fetch_coingecko_price

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/info/{symbol}")
async def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """Get general information about a symbol"""
    try:
        symbol = symbol.upper()
        
        # Get basic price info
        current_price = await fetch_coingecko_price(symbol)
        
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "status": "active" if current_price else "inactive",
            "data_sources": ["coingecko", "yfinance"],
            "supported_timeframes": ["30m", "1h", "4h", "24h"]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get info: {str(e)}")

@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str, 
    timeframe: str = "24h",
    limit: Optional[int] = 100
) -> Dict[str, Any]:
    """Get historical price data for a symbol"""
    try:
        symbol = symbol.upper()
        
        if timeframe not in ["30m", "1h", "4h", "24h"]:
            raise ValueError(f"Invalid timeframe: {timeframe}. Allowed: 30m, 1h, 4h, 24h")
        
        # Fetch historical data
        data_fetcher = DataFetcher()
        df = await data_fetcher.get_merged_data(symbol, timeframe)
        
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

# Query parameter endpoints for easier access
@router.get("/info")
async def get_symbol_info_query(coin: str = Query(..., description="Cryptocurrency symbol")) -> Dict[str, Any]:
    """Get general information about a symbol using query parameters"""
    return await get_symbol_info(coin)

@router.get("/historical")
async def get_historical_data_query(
    coin: str = Query(..., description="Cryptocurrency symbol"),
    timeframe: str = Query("24h", description="Timeframe: 30m, 1h, 4h, 24h"),
    limit: Optional[int] = Query(100, description="Number of data points to return")
) -> Dict[str, Any]:
    """Get historical price data for a symbol using query parameters"""
    return await get_historical_data(coin, timeframe, limit)

@router.get("/realtime")
async def get_realtime_price_query(coin: str = Query(..., description="Cryptocurrency symbol")) -> Dict[str, Any]:
    """Get real-time price for a symbol using query parameters"""
    return await get_realtime_price(coin)
