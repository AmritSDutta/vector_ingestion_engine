import asyncio
import logging

from celery import Task

from app.celery_worker import celery_app
from app.config.logging_config import setup_logging
from app.services.ingest_service import (
    ingest_and_store_embedding,
    ingest_and_store_to_all_database,
    prepare_ingestion_data,
    ingest_prepared_data_to_db
)

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


@celery_app.task(
    bind=True,
    max_retries=3,  # Increase retries
    autoretry_for=(Exception,),
    retry_backoff=True,  # Exponential backoff (1s, 2s, 4s, 8s...)
    retry_jitter=True  # Add randomness to prevent thundering herd)
)
def prepare_data_task(self: Task):
    logger.info("Starting prepare_data_task")

    def progress_callback(step: str):
        logger.info(f"Task status step: {step}")
        self.update_state(state='PROGRESS', meta={'step': step})

    return asyncio.run(prepare_ingestion_data(progress_callback))


@celery_app.task(bind=True,
                 max_retries=5,
                 autoretry_for=(Exception,),
                 retry_backoff=True,
                 retry_jitter=True)
def ingest_single_db_task(self: Task, filepath: str, db_name: str):
    logger.info(f"Starting ingest_single_db_task for {db_name}")

    def progress_callback(step: str):
        logger.info(f"Task status step: {step}")
        self.update_state(state='PROGRESS', meta={'step': step})

    return asyncio.run(ingest_prepared_data_to_db(filepath, db_name, progress_callback))
