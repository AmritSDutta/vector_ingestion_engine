import logging

from celery import Celery
from app.config.config import get_settings
from app.config.logging_config import setup_logging

settings = get_settings()
setup_logging()
logger = logging.getLogger(__name__)

# Singleton Celery instance configuration
celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['app.celery_task']  # IMPORTANT: Register the tasks
)

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True
)
