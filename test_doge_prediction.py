#!/usr/bin/env python3
import asyncio
import sys
import json
sys.path.append('.')
from controllers.prediction import predict_next_price

async def test_prediction():
    print('Testing DOGE 1h prediction with force retrain...')
    try:
        result = await predict_next_price('DOGE', '1h', force_retrain=True)
        print('SUCCESS! Full result JSON:')
        print(json.dumps(result, indent=2))
        
        print('\n=== SUMMARY ===')
        print(f'Symbol: {result["symbol"]}')
        print(f'Timeframe: {result["timeframe"]}')
        print(f'Predicted Price: ${result["predicted_price"]:.6f}')
        print(f'Last Price: ${result["last_price"]:.6f}')
        
        perf = result["model_performance"]
        print(f'\nModel Performance:')
        print(f'  R² Score: {perf["r2_score"]:.4f}')
        print(f'  MAPE: {perf["mape"]:.2f}%')
        print(f'  MAE: {perf["mae"]:.6f}')
        print(f'  RMSE: {perf["rmse"]:.6f}')
        print(f'  Trained on Demand: {perf["trained_on_demand"]}')
        
        arch = perf["model_architecture"]
        print(f'\nModel Architecture:')
        print(f'  Input Size: {arch["input_size"]}')
        print(f'  Hidden Size: {arch["hidden_size"]}')
        print(f'  Lookback: {arch["lookback"]}')
        
    except Exception as e:
        print(f'ERROR: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_prediction())
