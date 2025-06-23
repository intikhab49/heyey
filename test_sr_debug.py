#!/usr/bin/env python3
import asyncio
import sys
import logging
sys.path.append('.')

from controllers.prediction import predict_next_price, detect_support_resistance
from controllers.data_fetcher import DataFetcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_sr_debug():
    try:
        print("=== Testing Support/Resistance Debug ===")
        
        # Direct test of support/resistance detection
        print("\n1. Testing raw data input to support/resistance...")
        data_fetcher = DataFetcher()
        raw_data = await data_fetcher.get_merged_data('BTC', '24h')
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Raw Close price range: {raw_data['Close'].min():.2f} to {raw_data['Close'].max():.2f}")
        print(f"Last 3 Close prices: {raw_data['Close'].tail(3).values}")
        
        sr_result = detect_support_resistance(raw_data)
        print(f"\nSupport/Resistance result: {sr_result}")
        
        # Full prediction test
        print("\n2. Testing full prediction...")
        result = await predict_next_price('BTC', '24h')
        print(f"Prediction successful: {result.get('symbol')} {result.get('timeframe')}")
        print(f"Number of price levels: {len(result.get('price_levels', []))}")
        
        for level in result.get('price_levels', []):
            print(f"  {level['type'].upper()}: ${level['price']:.6f} (strength: {level['strength']:.3f})")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sr_debug())
