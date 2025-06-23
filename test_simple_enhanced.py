#!/usr/bin/env python3
"""
Simple test for enhanced training - test individual components
"""

import asyncio
import logging
from controllers.data_fetcher import DataFetcher
from controllers.enhanced_feature_processor import EnhancedFeatureProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_processing():
    """Test the enhanced data processing"""
    try:
        logger.info("🔍 Testing enhanced data processing")
        
        # Test data fetching
        data_fetcher = DataFetcher()
        df = await data_fetcher.get_merged_data("BTC", "24h")
        
        if df is None or df.empty:
            logger.error("❌ Failed to fetch data")
            return False
        
        logger.info(f"✅ Fetched data: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Test enhanced feature processing
        processor = EnhancedFeatureProcessor()
        
        # Check initial data
        logger.info(f"Initial data shape: {df.shape}")
        
        # Test correlation removal
        df_cleaned, scaler = processor.clean_and_scale_features(df.copy())
        
        logger.info(f"After processing shape: {df_cleaned.shape}")
        logger.info(f"Features removed: {len(processor.removed_features)}")
        logger.info(f"Final columns: {list(df_cleaned.columns)}")
        
        # Check if essential columns are preserved
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        preserved = [col for col in essential_cols if col in df_cleaned.columns]
        logger.info(f"Essential columns preserved: {preserved}")
        
        if len(preserved) >= 4:  # Need at least OHLC
            logger.info("✅ Enhanced data processing successful")
            return True
        else:
            logger.error("❌ Essential columns missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in data processing test: {str(e)}")
        return False

async def test_model_trainer():
    """Test the enhanced model trainer"""
    try:
        logger.info("🤖 Testing enhanced model trainer")
        
        from controllers.model_trainer import ModelTrainer
        
        # Get training data
        data_fetcher = DataFetcher()
        df = await data_fetcher.get_merged_data("BTC", "24h")
        
        if df is None or df.empty:
            logger.error("❌ No data for training")
            return False
        
        # Initialize trainer
        trainer = ModelTrainer("BTC", "24h")
        
        # Test enhanced training (with small epochs for testing)
        result = trainer.train_enhanced_model(df, epochs=5, patience=3)
        
        if result.get('success', False):
            logger.info(f"✅ Enhanced training successful")
            logger.info(f"   R² Score: {result.get('val_r2', 0):.4f}")
            logger.info(f"   MAPE: {result.get('val_mape', 0):.2f}%")
            logger.info(f"   Features selected: {result.get('features_selected', 0)}")
            return True
        else:
            logger.error("❌ Enhanced training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in model trainer test: {str(e)}")
        return False

async def main():
    """Run simple tests"""
    print("🧪 Simple Enhanced Training Test")
    print("=" * 50)
    
    # Test 1: Data processing
    data_test = await test_data_processing()
    
    if data_test:
        print("\n" + "="*50)
        # Test 2: Model training (only if data test passes)
        model_test = await test_model_trainer()
        
        if model_test:
            print("\n🎉 All tests passed! Enhanced training system is working.")
        else:
            print("\n⚠️ Model training test failed, but data processing works.")
    else:
        print("\n❌ Data processing test failed. Check the enhanced feature processor.")

if __name__ == "__main__":
    asyncio.run(main())
