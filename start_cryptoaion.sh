#!/bin/bash
# CryptoAion AI Startup Script with Fixed Environment

echo "🚀 Starting CryptoAion AI with fixes applied..."

# Navigate to project directory
cd /home/intikhab/CryptoAion-AI-main/Crypto-main/CryptoAion_AI_project_runnable

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source ../env/bin/activate

# Verify Python environment
echo "📍 Python path: $(which python3)"
echo "📦 Key packages:"
python3 -c "
import pandas, numpy, torch, cachetools, fastapi
print(f'✅ pandas: {pandas.__version__}')
print(f'✅ numpy: {numpy.__version__}') 
print(f'✅ torch: {torch.__version__}')
print(f'✅ cachetools: {cachetools.__version__}')
print(f'✅ fastapi: {fastapi.__version__}')
"

# Clear any corrupted cache
echo "🗑️ Clearing cache..."
rm -rf cache/ || true
rm -f *.pth || true

# Start the server
echo "🌐 Starting FastAPI server..."
python3 main.py
