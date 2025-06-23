#!/bin/bash
# Test API quickly
cd /home/intikhab/CryptoAion-AI-main/Crypto-main/CryptoAion_AI_project_runnable
source ../env/bin/activate

echo "Testing API endpoints..."
curl -s http://localhost:8000/ | head -5
echo ""
curl -s http://localhost:8000/health | head -5  
echo ""
curl -s http://localhost:8000/docs | head -10
