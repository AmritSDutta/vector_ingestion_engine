from fastapi import APIRouter

from app.routers import matching_doc_router, routes_query, routes_ingest

router = APIRouter()
router.include_router(matching_doc_router.doc_router)
router.include_router(routes_ingest.ingest_router)
router.include_router(routes_query.query_router)

