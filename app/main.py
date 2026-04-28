import logging
import subprocess
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends
from starlette.requests import Request

from app.config.config import get_settings
from app.config.logging_config import setup_logging
from app.dependencies.auth import get_current_username
from app.routers import app_router

# Initialize global logging before other imports
setup_logging()
logger = logging.getLogger(__name__)
app_name = get_settings().APP_NAME
port = get_settings().PORT


def _verify_tests_pass():
    subprocess.run(["pytest", "-q"], check=True)


@asynccontextmanager
async def lifespan(app_ins: FastAPI):
    logging.info(f'start: {app_ins.__str__()}')
    try:
        yield
    finally:
        logging.info('finish')


app = FastAPI(title=app_name, lifespan=lifespan)
app.include_router(app_router.router, prefix="/api",
                   dependencies=[Depends(get_current_username)])


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def health():
    logger.info('{"health": "Server in fine health"}')
    return {"health": "Server in fine health"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
