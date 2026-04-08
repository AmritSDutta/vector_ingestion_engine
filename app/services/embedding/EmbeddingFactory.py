import logging

from app.config.config import Settings
from app.services.embedding.MistralAIEmbeddingService import MistralAIEmbeddingService
from app.services.embedding.genai_service import GenAIEmbeddingService

logger = logging.getLogger(__name__)


def get_embedding_service():
    settings = Settings()
    
    # Check if primary embedder is genai and if its breaker is OPEN
    if settings.EMBEDDER == "genai":
        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBEDDING_DIM}')
        return GenAIEmbeddingService()
    elif settings.EMBEDDER == "mistralai":
        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBEDDING_DIM}')
        return MistralAIEmbeddingService()

    raise RuntimeError("unsupported embedder")
