from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import asyncio
import json
import logging
from typing import Dict, Any
from controllers.prediction import predict_next_price

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str, timeframe: str = "1h"):
    """WebSocket endpoint for real-time predictions - NO AUTHENTICATION"""
    await websocket.accept()
    
    try:
        logger.info(f"WebSocket connected for {symbol} {timeframe}")
        
        # Validate timeframe
        if timeframe not in ["30m", "1h", "4h", "24h"]:
            await websocket.send_text(json.dumps({
                "error": f"Invalid timeframe: {timeframe}. Allowed: 30m, 1h, 4h, 24h"
            }))
            await websocket.close()
            return
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "status": "connected",
            "symbol": symbol,
            "timeframe": timeframe,
            "message": f"Streaming predictions for {symbol} {timeframe}"
        }))
        
        # Main prediction loop
        while websocket.client_state == WebSocketState.CONNECTED:
            try:
                # Get prediction (load existing model or train if needed)
                prediction_result = await predict_next_price(symbol, timeframe, force_retrain=False)
                
                # Send prediction to client
                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "data": prediction_result,
                    "timestamp": "2025-06-20T16:30:00Z"
                }))
                
                # Wait before next prediction (30 seconds)
                await asyncio.sleep(30)
                
            except FileNotFoundError as e:
                # Handle case where model training fails
                error_msg = {
                    "type": "error",
                    "error": "Model training failed",
                    "message": f"Could not train model for {symbol} {timeframe}: {str(e)}",
                    "retrying_in": 60
                }
                await websocket.send_text(json.dumps(error_msg))
                await asyncio.sleep(60)  # Wait longer before retry
                
            except Exception as e:
                # Handle other prediction errors
                error_msg = {
                    "type": "error", 
                    "error": "Prediction failed",
                    "message": str(e),
                    "retrying_in": 30
                }
                await websocket.send_text(json.dumps(error_msg))
                await asyncio.sleep(30)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {symbol} {timeframe}")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol} {timeframe}: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "Internal server error",
                "message": str(e)
            }))
        except:
            pass  # Connection might be closed
    finally:
        logger.info(f"WebSocket connection closed for {symbol} {timeframe}")
