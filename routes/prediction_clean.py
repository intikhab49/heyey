from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from controllers.prediction import predict_next_price, get_model
from controllers.error_handler import error_tracker, ErrorSeverity, ErrorCategory
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/predict/{symbol}")
async def get_prediction(
    symbol: str, 
    timeframe: str = Query("1h", description="Timeframe: 30m, 1h, 4h, 24h"),
    force_retrain: bool = Query(False, description="Force model retraining")
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
        
        # Get prediction directly
        result = await predict_next_price(symbol, timeframe, force_retrain=force_retrain)
        
        if result is None:
            raise HTTPException(
                status_code=400, 
                detail=f"Unable to generate prediction for {symbol} {timeframe}. Model training may have failed."
            )
        
        # Add success metadata
        result.update({
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "success"
        })
        
        logger.info(f"Prediction successful for {symbol} {timeframe}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error for {symbol} {timeframe}: {e}")
        error_tracker.log_error(str(e), ErrorSeverity.LOW, ErrorCategory.VALIDATION)
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Prediction error for {symbol} {timeframe}: {e}")
        error_tracker.log_error(str(e), ErrorSeverity.HIGH, ErrorCategory.PREDICTION)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal error during prediction: {str(e)}"
        )

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
        model_info = await get_model(symbol, timeframe)
        
        if model_info is None:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_exists": False,
                "message": f"No trained model found for {symbol} {timeframe}. Use force_retrain=true to train."
            }
        
        # Try to get prediction without retraining
        result = await predict_next_price(symbol, timeframe, force_retrain=False)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_exists": True,
            "prediction_available": result is not None,
            "model_metrics": model_info.get("metrics", {}),
            "message": "Model ready for predictions" if result else "Model exists but prediction failed"
        }
        
    except Exception as e:
        logger.error(f"Test prediction error for {symbol} {timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "crypto-prediction-api",
        "endpoints": [
            "/predict/{symbol}",
            "/test-predict/{symbol}",
            "/health"
        ]
    }
