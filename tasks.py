"""Background tasks for model training and predictions"""

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from celery import current_task
from celery_app import celery_app
from controllers.prediction import get_model, predict_next_price
from controllers.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def train_and_predict_task(self, symbol: str, timeframe: str = "1h", force_retrain: bool = False) -> Dict[str, Any]:
    """
    Background task to train a model and make a prediction
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        timeframe: Time interval ('30m', '1h', '4h', '24h')
        force_retrain: Whether to force retraining even if model exists
        
    Returns:
        Dictionary with prediction results and model metrics
    """
    task_id = self.request.id
    
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 10,
                'total': 100,
                'status': f'Starting training for {symbol} {timeframe}...'
            }
        )
        
        # Run the async prediction function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                _async_train_and_predict(self, symbol, timeframe, force_retrain)
            )
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Task {task_id} failed: {str(exc)}")
        logger.error(traceback.format_exc())
        
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(exc),
                'traceback': traceback.format_exc(),
                'symbol': symbol,
                'timeframe': timeframe
            }
        )
        raise exc

async def _async_train_and_predict(task, symbol: str, timeframe: str, force_retrain: bool) -> Dict[str, Any]:
    """Internal async function to handle the actual training and prediction"""
    
    try:
        # Update progress
        task.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'status': f'Checking data availability for {symbol}...'
            }
        )
        
        # Check if we need to train
        model_info = await get_model(symbol, timeframe)
        needs_training = force_retrain or model_info is None
        
        if needs_training:
            task.update_state(
                state='PROGRESS',
                meta={
                    'current': 50,
                    'total': 100,
                    'status': f'Training model for {symbol} {timeframe}... This may take several minutes.'
                }
            )
        
        # Update progress
        task.update_state(
            state='PROGRESS',
            meta={
                'current': 80,
                'total': 100,
                'status': f'Making prediction for {symbol}...'
            }
        )
        
        # Make prediction (this will train if needed)
        result = await predict_next_price(symbol, timeframe, force_retrain)
        
        # Update progress
        task.update_state(
            state='PROGRESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Prediction completed successfully!'
            }
        )
        
        # Add task metadata
        result.update({
            'task_id': task.request.id,
            'completed_at': datetime.utcnow().isoformat(),
            'trained_on_demand': needs_training
        })
        
        return result
        
    except Exception as exc:
        logger.error(f"Error in async training/prediction: {str(exc)}")
        raise exc

@celery_app.task(bind=True)
def get_task_status(self, task_id: str) -> Dict[str, Any]:
    """Get the status of a background task"""
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            response = {
                'state': result.state,
                'status': 'Task is waiting to be processed...'
            }
        elif result.state == 'PROGRESS':
            response = {
                'state': result.state,
                'current': result.info.get('current', 0),
                'total': result.info.get('total', 100),
                'status': result.info.get('status', 'Processing...')
            }
        elif result.state == 'SUCCESS':
            response = {
                'state': result.state,
                'result': result.result
            }
        else:  # FAILURE
            response = {
                'state': result.state,
                'error': str(result.info),
            }
        
        return response
        
    except Exception as exc:
        logger.error(f"Error getting task status: {str(exc)}")
        return {
            'state': 'FAILURE',
            'error': f'Could not retrieve task status: {str(exc)}'
        }
