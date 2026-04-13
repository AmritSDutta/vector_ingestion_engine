import logging

logger = logging.getLogger(__name__)


async def time_coro(label, coro):
    import time
    start = time.perf_counter()
    result = await coro
    logging.info(f"{label} took: {time.perf_counter() - start:.4f}s")
    return result
