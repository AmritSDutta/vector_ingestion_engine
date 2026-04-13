from fastapi import APIRouter

from app.routers import routes_query, routes_ingest

router = APIRouter()
router.include_router(routes_ingest.ingest_router)
router.include_router(routes_query.query_router)
