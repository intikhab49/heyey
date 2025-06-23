#!/usr/bin/env python3
import asyncio
import sys
import json
sys.path.append('.')
from controllers.prediction import predict_next_price

async def test_24h_prediction():
    print('Testing DOGE 24h prediction with force retrain...')
    try:
        result = await predict_next_price('DOGE', '24h', force_retrain=True)
        print('SUCCESS! Summary:')
        
        perf = result["model_performance"]
        print(f'\nModel Performance for 24h:')
        print(f'  R² Score: {perf["r2_score"]:.4f}')
        print(f'  MAPE: {perf["mape"]:.2f}%')
        print(f'  MAE: {perf["mae"]:.6f}')
        print(f'  RMSE: {perf["rmse"]:.6f}')
        print(f'  Trained on Demand: {perf["trained_on_demand"]}')
        
        print(f'\nPrediction Results:')
        print(f'  Symbol: {result["symbol"]}')
        print(f'  Predicted Price: ${result["predicted_price"]:.6f}')
        print(f'  Last Price: ${result["last_price"]:.6f}')
        print(f'  Price Change: {((result["predicted_price"] / result["last_price"]) - 1) * 100:.2f}%')
        
        # Test if R² is now positive and good
        if perf["r2_score"] > 0.7:
            print(f'✅ EXCELLENT: R² = {perf["r2_score"]:.4f} (> 0.7)')
        elif perf["r2_score"] > 0.5:
            print(f'✅ GOOD: R² = {perf["r2_score"]:.4f} (> 0.5)')
        elif perf["r2_score"] > 0.3:
            print(f'⚠️  DECENT: R² = {perf["r2_score"]:.4f} (> 0.3)')
        elif perf["r2_score"] > 0:
            print(f'⚠️  WEAK: R² = {perf["r2_score"]:.4f} (> 0)')
        else:
            print(f'❌ POOR: R² = {perf["r2_score"]:.4f} (< 0)')
        
        if perf["mape"] < 5:
            print(f'✅ EXCELLENT: MAPE = {perf["mape"]:.2f}% (< 5%)')
        elif perf["mape"] < 10:
            print(f'✅ GOOD: MAPE = {perf["mape"]:.2f}% (< 10%)')
        elif perf["mape"] < 15:
            print(f'⚠️  DECENT: MAPE = {perf["mape"]:.2f}% (< 15%)')
        else:
            print(f'❌ POOR: MAPE = {perf["mape"]:.2f}% (> 15%)')
        
    except Exception as e:
        print(f'ERROR: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_24h_prediction())
