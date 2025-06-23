#!/usr/bin/env python3
"""
Quick test to check if enhanced training method is accessible
"""

import asyncio
import logging

async def test_method_access():
    """Test if the enhanced training method can be accessed"""
    try:
        from controllers.model_trainer import ModelTrainer
        
        # Create trainer instance
        trainer = ModelTrainer("BTC", "24h")
        
        # Check if method exists
        if hasattr(trainer, 'train_enhanced_model'):
            print("✅ train_enhanced_model method exists")
            return True
        else:
            print("❌ train_enhanced_model method not found")
            # Show available methods
            methods = [method for method in dir(trainer) if not method.startswith('_')]
            print(f"Available methods: {methods}")
            return False
    
    except Exception as e:
        print(f"❌ Error accessing ModelTrainer: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_method_access())
