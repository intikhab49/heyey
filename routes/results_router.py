"""Results router for checking background task status and retrieving predictions"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from tasks import get_task_status
from celery_app import celery_app
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])

@router.get("/{task_id}")
async def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    Get the status and result of a background prediction task
    
    Args:
        task_id: The unique task identifier returned from the prediction request
        
    Returns:
        Task status and prediction result if completed
    """
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            return {
                'task_id': task_id,
                'state': result.state,
                'status': 'Task is waiting to be processed...',
                'progress': {'current': 0, 'total': 100}
            }
        elif result.state == 'PROGRESS':
            return {
                'task_id': task_id,
                'state': result.state,
                'status': result.info.get('status', 'Processing...'),
                'progress': {
                    'current': result.info.get('current', 0),
                    'total': result.info.get('total', 100)
                }
            }
        elif result.state == 'SUCCESS':
            return {
                'task_id': task_id,
                'state': result.state,
                'result': result.result,
                'status': 'Prediction completed successfully!'
            }
        elif result.state == 'FAILURE':
            error_info = result.info if result.info else 'Unknown error occurred'
            return {
                'task_id': task_id,
                'state': result.state,
                'error': str(error_info),
                'status': 'Task failed'
            }
        else:
            return {
                'task_id': task_id,
                'state': result.state,
                'status': f'Task in state: {result.state}'
            }
            
    except Exception as exc:
        logger.error(f"Error retrieving task {task_id}: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving task status: {str(exc)}"
        )

@router.get("/")
async def list_active_tasks() -> Dict[str, Any]:
    """
    List all active tasks in the system
    
    Returns:
        Information about active tasks
    """
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return {
                'active_tasks': [],
                'total_active': 0,
                'message': 'No active tasks found'
            }
        
        # Format the response
        tasks_info = []
        total_tasks = 0
        
        for worker, tasks in active_tasks.items():
            for task in tasks:
                tasks_info.append({
                    'task_id': task.get('id'),
                    'name': task.get('name'),
                    'worker': worker,
                    'args': task.get('args', []),
                    'kwargs': task.get('kwargs', {}),
                })
                total_tasks += 1
        
        return {
            'active_tasks': tasks_info,
            'total_active': total_tasks,
            'workers': list(active_tasks.keys())
        }
        
    except Exception as exc:
        logger.error(f"Error listing active tasks: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tasks: {str(exc)}"
        )
