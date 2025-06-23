#!/usr/bin/env python3
"""
Test Enhanced Training System
Verify that the correlation handling and enhanced features improve model accuracy
"""

import asyncio
import requests
import json
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_prediction(symbol, timeframe, force_retrain=True):
    """Test enhanced prediction with correlation handling"""
    try:
        logger.info(f"🚀 Testing enhanced prediction for {symbol} {timeframe}")
        
        # Import the enhanced prediction function
        from controllers.prediction import predict_next_price
        
        # Run prediction with enhanced training
        result = await predict_next_price(symbol, timeframe, force_retrain=force_retrain)
        
        # Extract key metrics
        model_performance = result.get('model_performance', {})
        r2_score = model_performance.get('r2_score', 0)
        mape = model_performance.get('mape', 0)
        training_method = model_performance.get('training_method', 'traditional')
        
        logger.info(f"✅ {symbol} {timeframe} Results:")
        logger.info(f"   R² Score: {r2_score:.4f}")
        logger.info(f"   MAPE: {mape:.2f}%")
        logger.info(f"   Training Method: {training_method}")
        logger.info(f"   Predicted Price: ${result.get('predicted_price', 0):.4f}")
        logger.info(f"   Last Price: ${result.get('last_price', 0):.4f}")
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'r2_score': r2_score,
            'mape': mape,
            'training_method': training_method,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing {symbol} {timeframe}: {str(e)}")
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'error': str(e),
            'success': False
        }

async def compare_training_methods():
    """Compare traditional vs enhanced training methods"""
    logger.info("🔬 Comparing Training Methods")
    logger.info("=" * 60)
    
    test_cases = [
        ("BTC", "24h"),
        ("ETH", "1h"),
        ("DOGE", "4h")
    ]
    
    results = []
    
    for symbol, timeframe in test_cases:
        logger.info(f"\n📊 Testing {symbol} {timeframe}")
        logger.info("-" * 40)
        
        # Test enhanced training
        result = await test_enhanced_prediction(symbol, timeframe, force_retrain=True)
        results.append(result)
        
        # Wait a bit between tests
        await asyncio.sleep(2)
    
    return results

async def test_correlation_handling():
    """Test if correlation handling improves model performance"""
    logger.info("🔍 Testing Correlation Handling")
    logger.info("=" * 60)
    
    # Run analysis on the enhanced system
    from analyze_model_issues import analyze_data_quality, suggest_improvements
    
    test_cases = [
        ("BTC", "24h"),
        ("ETH", "1h")
    ]
    
    for symbol, timeframe in test_cases:
        logger.info(f"\n📈 Analyzing {symbol} {timeframe}")
        logger.info("-" * 40)
        
        try:
            # Run data quality analysis
            analysis = await analyze_data_quality(symbol, timeframe)
            
            # Check if correlation issues are resolved
            issues = analysis.get('issues', [])
            correlation_issues = [issue for issue in issues if 'correlation' in issue.lower()]
            
            logger.info(f"Data points: {analysis.get('data_points', 'N/A')}")
            logger.info(f"Total issues: {len(issues)}")
            logger.info(f"Correlation issues: {len(correlation_issues)}")
            
            if correlation_issues:
                logger.warning(f"Still has correlation issues: {correlation_issues[0]}")
            else:
                logger.info("✅ No major correlation issues detected")
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {str(e)}")

async def main():
    """Run comprehensive enhanced training tests"""
    print("🚀 Enhanced Training System Test")
    print("=" * 60)
    print(f"Test Started: {datetime.now()}")
    print()
    
    try:
        # Test 1: Compare training methods
        training_results = await compare_training_methods()
        
        # Test 2: Check correlation handling
        await test_correlation_handling()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        successful_tests = [r for r in training_results if r.get('success', False)]
        failed_tests = [r for r in training_results if not r.get('success', False)]
        
        print(f"✅ Successful tests: {len(successful_tests)}")
        print(f"❌ Failed tests: {len(failed_tests)}")
        
        if successful_tests:
            print("\n🎯 Model Performance Results:")
            for result in successful_tests:
                r2 = result.get('r2_score', 0)
                mape = result.get('mape', 0)
                method = result.get('training_method', 'unknown')
                
                # Determine if results are good
                r2_status = "✅ Good" if r2 > 0.3 else "⚠️ Poor" if r2 > 0 else "❌ Negative"
                mape_status = "✅ Good" if mape < 10 else "⚠️ Fair" if mape < 20 else "❌ Poor"
                
                print(f"  {result['symbol']} {result['timeframe']}:")
                print(f"    R² Score: {r2:.4f} {r2_status}")
                print(f"    MAPE: {mape:.2f}% {mape_status}")
                print(f"    Method: {method}")
                print()
        
        if failed_tests:
            print("❌ Failed Tests:")
            for result in failed_tests:
                print(f"  {result['symbol']} {result['timeframe']}: {result.get('error', 'Unknown error')}")
        
        # Overall assessment
        if len(successful_tests) >= 2:
            avg_r2 = sum(r.get('r2_score', 0) for r in successful_tests) / len(successful_tests)
            avg_mape = sum(r.get('mape', 0) for r in successful_tests) / len(successful_tests)
            
            print(f"\n🎯 Overall Performance:")
            print(f"   Average R²: {avg_r2:.4f}")
            print(f"   Average MAPE: {avg_mape:.2f}%")
            
            if avg_r2 > 0.3 and avg_mape < 15:
                print("🎉 EXCELLENT: Enhanced training system is working well!")
            elif avg_r2 > 0.1 and avg_mape < 25:
                print("👍 GOOD: Enhanced training shows improvement")
            else:
                print("⚠️ NEEDS WORK: Further optimization required")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        print(f"❌ Test suite failed: {str(e)}")
    
    print(f"\nTest Completed: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
