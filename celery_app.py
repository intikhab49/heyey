"""Celery configuration for background task processing"""

from celery import Celery
import os
from simple_config import settings

# Create Celery app
celery_app = Celery(
    "cryptoaion",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["tasks"]
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

if __name__ == "__main__":
    celery_app.start()
