import asyncio
import websockets
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_websocket(symbol="BTC-USD", timeframe="1h"):
    """Test WebSocket connection for cryptocurrency updates."""
    uri = f"ws://localhost:8000/ws/{symbol}?timeframe={timeframe}"
    
    try:
        logger.info(f"Connecting to WebSocket for {symbol} with {timeframe} timeframe...")
        async with websockets.connect(uri) as websocket:
            logger.info("WebSocket connection established!")
            
            # Listen for messages for 60 seconds
            for _ in range(6):  # 10 seconds * 6 = 60 seconds
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    logger.info(f"Received data: {json.dumps(data, indent=2)}")
                except asyncio.TimeoutError:
                    logger.warning("No message received in 10 seconds")
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {message}")
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket connection closed unexpectedly: {e}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

if __name__ == "__main__":
    # Test different symbols and timeframes
    test_cases = [
        ("BTC-USD", "1h"),
        ("ETH-USD", "30m"),
        ("BNB-USD", "4h")
    ]
    
    for symbol, timeframe in test_cases:
        logger.info(f"\nTesting {symbol} with {timeframe} timeframe")
        asyncio.get_event_loop().run_until_complete(test_websocket(symbol, timeframe))
        logger.info(f"Completed test for {symbol}") 