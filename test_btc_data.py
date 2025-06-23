#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controllers.data_fetcher import DataFetcher
import simple_config
import logging

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

def test_btc_data():
    print("Testing BTC data fetching with MultiIndex fix...")
    
    data_fetcher = DataFetcher()
    
    # Test BTC 30m data
    print("\n=== Testing BTC 30m ===")
    btc_30m = data_fetcher.fetch_yfinance_data('BTC', '30m')
    
    if btc_30m is not None and not btc_30m.empty:
        print(f"BTC 30m data shape: {btc_30m.shape}")
        print(f"Columns: {list(btc_30m.columns)}")
        print(f"Close column range: {btc_30m['Close'].min():.2f} to {btc_30m['Close'].max():.2f}")
        print(f"Sample of last 3 rows:")
        print(btc_30m[['Open', 'High', 'Low', 'Close', 'Volume']].tail(3))
    else:
        print("Failed to fetch BTC 30m data")
    
    print("\n=== Testing BTC 1h ===")
    btc_1h = data_fetcher.fetch_yfinance_data('BTC', '1h')
    
    if btc_1h is not None and not btc_1h.empty:
        print(f"BTC 1h data shape: {btc_1h.shape}")
        print(f"Columns: {list(btc_1h.columns)}")
        print(f"Close column range: {btc_1h['Close'].min():.2f} to {btc_1h['Close'].max():.2f}")
        print(f"Sample of last 3 rows:")
        print(btc_1h[['Open', 'High', 'Low', 'Close', 'Volume']].tail(3))
    else:
        print("Failed to fetch BTC 1h data")

if __name__ == "__main__":
    test_btc_data()
