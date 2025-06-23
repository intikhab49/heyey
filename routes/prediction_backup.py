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
            }
        
        # Otherwise, run synchronously (for existing models or when background=False)
        result = await predict_next_price(symbol, timeframe, force_retrain)
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.get('message', 'Unknown error')}"
            )
            
        result["background"] = False
        return result
        
    except ValueError as e:
        error_id = error_tracker.track_error(
            e,
            ErrorSeverity.ERROR,
            ErrorCategory.API,
            "prediction_endpoint"
        )
        logger.error(f"Prediction error (Error ID: {error_id}): {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        error_id = error_tracker.track_error(
            e,
            ErrorSeverity.ERROR,
            ErrorCategory.API,
            "prediction_endpoint"
        )
        logger.error(f"Unexpected error (Error ID: {error_id}): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/predict")
async def post_prediction(data: dict) -> Dict[str, Any]:
    """
    Alternative POST endpoint for predictions
    
    Args:
        data: Dictionary with 'symbol', 'timeframe', optional 'force_retrain' and 'background'
        
    Returns:
        Prediction result or background task information
    """
    try:
        symbol = data.get("symbol")
        timeframe = data.get("timeframe", "1h")
        force_retrain = data.get("force_retrain", False)
        background = data.get("background", True)
        
        if not symbol:
            raise ValueError("Missing symbol in request body")
        
        # Use the same logic as GET endpoint
        return await get_prediction(symbol, timeframe, force_retrain, background)
        
    except Exception as e:
        logger.error(f"Error in POST prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-predict")
async def test_prediction() -> Dict[str, Any]:
    """Quick test endpoint with BTC prediction"""
    try:
        result = await predict_next_price("BTC", "1h", force_retrain=False)
        return {
            "test": "success",
            "result": result,
            "message": "Test prediction completed"
        }
    except Exception as e:
        logger.error(f"Test prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")