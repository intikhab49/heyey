from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from controllers.prediction import predict_next_price, get_model
from controllers.error_handler import error_tracker, ErrorSeverity, ErrorCategory
import logging
import numpy as np

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
    else:
        return obj

@router.get("/predict/{symbol}")
async def get_prediction(
    symbol: str, 
    timeframe: str = Query("1h", description="Timeframe: 30m, 1h, 4h, 24h"),
    force_retrain: bool = Query(True, description="Force model retraining")
) -> Dict[str, Any]:
    """
    Get price prediction for a cryptocurrency
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC, ETH, DOGE)
        timeframe: Time interval (30m, 1h, 4h, 24h)
        force_retrain: Whether to force model retraining
        
    Returns:
        Prediction result with metrics and confidence
    """
    try:
        # Validate inputs
        symbol = symbol.upper()
        if timeframe not in ["30m", "1h", "4h", "24h"]:
            raise ValueError(f"Invalid timeframe: {timeframe}. Allowed: 30m, 1h, 4h, 24h")
        
        logger.info(f"Prediction request: {symbol} {timeframe}, force_retrain={force_retrain}")
        
        # Get prediction with force_retrain parameter
        result = await predict_next_price(symbol, timeframe, force_retrain=force_retrain)
        
        if result is None:
            raise HTTPException(
                status_code=400, 
                detail=f"Unable to generate prediction for {symbol} {timeframe}. Model training may have failed."
            )
        
        # Convert numpy types to Python types for JSON serialization
        result = convert_numpy_types(result)
        
        # Use existing price_levels from the result, or extract from confidence factors as fallback
        existing_price_levels = result.get('price_levels', [])
        confidence_factors = result.get('confidence_factors', {})
        price_levels_from_confidence = confidence_factors.get('price_levels', [])
        
        # Prefer existing price_levels, fall back to confidence_factors if empty
        price_levels = existing_price_levels if existing_price_levels else []
        
        # If no price levels found, try extracting from confidence factors
        if not price_levels and price_levels_from_confidence:
            for level in price_levels_from_confidence:
                if isinstance(level, dict) and 'price' in level:
                    price_levels.append({
                        'type': level.get('type', 'unknown'),
                        'price': float(level['price']),
                        'strength': float(level.get('strength', 0.0)),
                        'confidence': float(level.get('confidence', 0.0)),
                        'signal': level.get('signal', 'neutral')
                    })
        
        # Add success metadata and ensure price levels are included
        result.update({
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "success",
            "price_levels": price_levels
        })
        
        logger.info(f"Prediction successful for {symbol} {timeframe}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error for {symbol} {timeframe}: {e}")
        error_tracker.track_error(e, ErrorSeverity.WARNING, ErrorCategory.API, "prediction_router")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Prediction error for {symbol} {timeframe}: {e}")
        error_tracker.track_error(e, ErrorSeverity.ERROR, ErrorCategory.API, "prediction_router")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal error during prediction: {str(e)}"
        )

@router.get("/predict")
async def get_prediction_query(
    coin: str = Query(..., description="Cryptocurrency symbol (e.g., BTC, ETH, DOGE)"),
    timeframe: str = Query("1h", description="Timeframe: 30m, 1h, 4h, 24h"),
    force_retrain: bool = Query(True, description="Force model retraining")
) -> Dict[str, Any]:
    """
    Get price prediction for a cryptocurrency using query parameters
    
    Args:
        coin: Cryptocurrency symbol (e.g., BTC, ETH, DOGE)
        timeframe: Time interval (30m, 1h, 4h, 24h)
        force_retrain: Whether to force model retraining
        
    Returns:
        Prediction result with metrics and confidence
    """
    # Delegate to the main prediction function
    return await get_prediction(coin, timeframe, force_retrain)

@router.get("/test-predict/{symbol}")
async def test_prediction(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 30m, 1h, 4h, 24h")
) -> Dict[str, Any]:
    """
    Test endpoint to check if a model exists and can make predictions
    """
    try:
        symbol = symbol.upper()
        if timeframe not in ["30m", "1h", "4h", "24h"]:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Check if model exists
        try:
            model_info = await get_model(symbol, timeframe)
            model_exists = model_info is not None
            training_metrics = model_info[4] if model_info and len(model_info) > 4 else {}
        except Exception:
            model_exists = False
            training_metrics = {}
        
        if not model_exists:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_exists": False,
                "message": f"No trained model found for {symbol} {timeframe}. Use force_retrain=true to train."
            }
        
        # Try to get prediction
        try:
            result = await predict_next_price(symbol, timeframe)
            prediction_available = result is not None
        except Exception:
            prediction_available = False
        
        return convert_numpy_types({
            "symbol": symbol,
            "timeframe": timeframe,
            "model_exists": True,
            "prediction_available": prediction_available,
            "model_metrics": training_metrics,
            "message": "Model ready for predictions" if prediction_available else "Model exists but prediction failed"
        })
        
    except Exception as e:
        logger.error(f"Test prediction error for {symbol} {timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


