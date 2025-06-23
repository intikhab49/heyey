#!/bin/bash
# Quick dependency fix
cd /home/intikhab/CryptoAion-AI-main/Crypto-main
source env/bin/activate
cd CryptoAion_AI_project_runnable
pip install cachetools pandas numpy torch scikit-learn ta fastapi uvicorn
echo "✅ Dependencies fixed"
