import os
import sys
import logging
import argparse
import asyncio
from typing import List, Dict
from controllers.prediction import get_model
from simple_config import settings
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Supported cryptocurrencies
SUPPORTED_COINS = [
    "BTC",  # Bitcoin
    "ETH",  # Ethereum
    "BNB",  # Binance Coin
    "XRP",  # Ripple
    "ADA",  # Cardano
    "DOGE", # Dogecoin
    "SOL",  # Solana
    "NEAR"  # NEAR Protocol
]

async def train_model(symbol: str, timeframe: str) -> Dict:
    """Train model for a specific symbol and timeframe"""
    max_retries = 3
    retry_delay = 60  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Training model for {symbol} ({timeframe}) - Attempt {attempt + 1}/{max_retries}")
            
            # Use get_model with force_retrain=True to train a fresh model
            model, feature_scaler, target_scaler, feature_names = await get_model(symbol, timeframe, force_retrain=True)
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'timeframe': timeframe,
                'feature_count': len(feature_names),
                'attempt': attempt + 1
            }
            
            logger.info(f"Successfully trained model for {symbol} ({timeframe})")
            logger.info(f"Features used: {len(feature_names)}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error training model for {symbol} ({timeframe}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                continue
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'error',
                'error': str(e)
            }
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'status': 'error',
        'error': f"Failed after {max_retries} attempts"
    }

async def train_all_models(symbols: List[str] = None, timeframes: List[str] = None) -> List[Dict]:
    """Train models for all symbols and timeframes"""
    if symbols is None:
        symbols = SUPPORTED_COINS
    if timeframes is None:
        timeframes = ["30m", "1h", "4h", "24h"]
        
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    results = []
    total_models = len(symbols) * len(timeframes)
    current_model = 0
    
    logger.info(f"Starting training for {total_models} models")
    
    for symbol in symbols:
        for timeframe in timeframes:
            current_model += 1
            logger.info(f"Training model {current_model}/{total_models}: {symbol} {timeframe}")
            
            start_time = time.time()
            result = await train_model(symbol, timeframe)
            end_time = time.time()
            
            result['training_time'] = end_time - start_time
            results.append(result)
            
            logger.info(f"Completed {symbol} {timeframe} in {result['training_time']:.2f}s")
    
    # Summary
    successful = len([r for r in results if r.get('status') == 'success'])
    failed = len(results) - successful
    
    logger.info(f"Training complete: {successful} successful, {failed} failed")
    
    return results

async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train cryptocurrency prediction models')
    parser.add_argument('--symbol', type=str, help='Specific symbol to train (e.g., BTC)')
    parser.add_argument('--timeframe', type=str, choices=['30m', '1h', '4h', '24h'], 
                       help='Specific timeframe to train')
    parser.add_argument('--all', action='store_true', help='Train all supported models')
    
    args = parser.parse_args()
    
    if args.symbol and args.timeframe:
        # Train specific model
        result = await train_model(args.symbol, args.timeframe)
        if result.get('status') == 'success':
            logger.info("Training completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"Training failed: {result.get('error')}")
            sys.exit(1)
            
    elif args.all:
        # Train all models
        results = await train_all_models()
        successful = len([r for r in results if r.get('status') == 'success'])
        total = len(results)
        
        if successful == total:
            logger.info("All models trained successfully!")
            sys.exit(0)
        else:
            logger.error(f"Training completed with {total - successful} failures")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())