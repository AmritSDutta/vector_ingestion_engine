import asyncio
import logging
from celery import Task
from app.celery_worker import celery_app
from app.config.logging_config import setup_logging
from app.services.ingest_service import ingest_and_store_embedding, ingest_and_store_to_all_database

logger = logging.getLogger(__name__)
setup_logging()


@celery_app.task(bind=True)
def ingest_task_wrapper(self: Task):
    """
    Synchronous wrapper for the async ingestion logic.
    Provides a callback for updating progress in Redis.
    """
    logger.info("Starting ingest_task_wrapper")

    def progress_callback(step: str):
        logger.info(f"Task status step: {step}")
        self.update_state(state='PROGRESS', meta={'step': step})

    return asyncio.run(ingest_and_store_embedding(progress_callback))


@celery_app.task(bind=True)
def ingest_all_db_task_wrapper(self: Task):
    """
    Synchronous wrapper for the async ingestion logic.
    Provides a callback for updating progress in Redis.
    """
    logger.info("Starting ingest_task_wrapper")

    def progress_callback(step: str):
        logger.info(f"Task status step: {step}")
        self.update_state(state='PROGRESS', meta={'step': step})

    return asyncio.run(ingest_and_store_to_all_database(progress_callback))
