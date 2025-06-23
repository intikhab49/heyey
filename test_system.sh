#!/bin/bash
# CryptoAion AI Test Script

cd /home/intikhab/CryptoAion-AI-main/Crypto-main/CryptoAion_AI_project_runnable
source ../env/bin/activate

echo "🧪 Testing CryptoAion AI System..."

# Test 1: Technical Indicators
echo "1️⃣ Testing technical indicators..."
python3 -c "
from controllers.feature_engineering import calculate_technical_indicators
import pandas as pd
import numpy as np

# Create test data
df = pd.DataFrame({
    'Close': [50000, 50500, 49800, 51000, 50200],
    'High': [50200, 50700, 50000, 51200, 50400],
    'Low': [49800, 50300, 49500, 50800, 50000],
    'Open': [50000, 50000, 50500, 49800, 51000],
    'Volume': [1000000] * 5
})

result = calculate_technical_indicators(df)
rsi = result['RSI_14'].dropna()
if len(rsi) > 0 and all(0 <= x <= 100 for x in rsi):
    print('✅ Technical indicators working correctly')
else:
    print('❌ Technical indicators have issues')
"

# Test 2: Environment Check  
echo "2️⃣ Testing environment..."
python3 -c "
try:
    import cachetools, pandas, numpy, torch, fastapi
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Missing package: {e}')
"

# Test 3: Config Check
echo "3️⃣ Testing configuration..."
python3 -c "
try:
    from simple_config import TIMEFRAME_MAP, settings
    print(f'✅ Config loaded. Timeframes: {list(TIMEFRAME_MAP.keys())}')
except Exception as e:
    print(f'❌ Config error: {e}')
"

echo "🎯 Test complete! Check output above for any ❌ errors."
