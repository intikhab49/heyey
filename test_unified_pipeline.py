import asyncio
import logging
import sys
import os
import argparse

# Add project root to sys.path to allow imports from controllers, config, etc.
# This assumes the script is in the project root or a subdirectory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from controllers.prediction import get_model, predict_next_price
    from simple_config import settings # To get lookback from TIMEFRAME_MAP
except ImportError as e:
    print(f"Error importing necessary modules. Ensure you are in the project root directory an PYTHONPATH is set up if needed: {e}")
    sys.exit(1)

# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for very verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def run_test(symbol: str = "BTC", timeframe: str = "1h"):
    """Run test for a specific symbol and timeframe"""
    try:
        logger.info(f"--- Starting Test for {symbol} - {timeframe} ---")
        
        # Phase 1: Model Training (Always Fresh)
        logger.info("Phase 1: Training Fresh Model")
        model_tuple = await get_model(symbol, timeframe, force_retrain=True)
        if model_tuple is None:
            logger.error("Failed to get model")
            return False
            
        # Unpack the model tuple
        model, feature_scaler, target_scaler, feature_names = model_tuple
            
        # Phase 2: Price Prediction
        logger.info("Phase 2: Making Prediction")
        prediction_result = await predict_next_price(symbol, timeframe)
        if prediction_result is None:
            logger.error("Failed to get prediction")
            return False
            
        logger.info(f"Successfully received prediction result for {symbol} - {timeframe}:")
        logger.info(prediction_result)
        logger.info(f"--- Test Finished for {symbol} - {timeframe} ---")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Test the unified prediction pipeline")
    parser.add_argument("--symbol", type=str, default="BTC", help="Symbol to test (default: BTC)")
    parser.add_argument("--timeframe", type=str, default="1h", choices=settings.TIMEFRAME_OPTIONS,
                      help=f"Timeframe to test (default: 1h, options: {', '.join(settings.TIMEFRAME_OPTIONS)})")
    return parser.parse_args()

if __name__ == "__main__":
    # Ensure event loop is managed correctly if running from a non-async entry point
    if sys.version_info >= (3, 7) and sys.platform.startswith('win') and \
       isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
        # For Windows + Python 3.7/3.8, Proactor event loop might be needed for subprocesses.
        # Not strictly necessary here but good practice if other async ops were involved.
        pass # asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        args = parse_args()
        
        # Run the test
        success = asyncio.run(run_test(args.symbol, args.timeframe))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test script interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled exception in test_unified_pipeline.py: {e}", exc_info=True) 