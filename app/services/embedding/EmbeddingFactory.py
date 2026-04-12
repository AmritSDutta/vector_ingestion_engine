import logging

from app.config.config import Settings, get_settings
from app.services.embedding.MistralAIEmbeddingService import MistralAIEmbeddingService
from app.services.embedding.genai_service import GenAIEmbeddingService

logger = logging.getLogger(__name__)

_genAIEmbeddingService: GenAIEmbeddingService | None = None
_mistralAIEmbeddingService: MistralAIEmbeddingService | None = None


def get_embedding_service():
    settings = get_settings()
    logging.info(f'Embedding  model: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBEDDING_DIM}')

    global _genAIEmbeddingService, _mistralAIEmbeddingService
    # Check if primary embedder is genai and if its breaker is OPEN
    if settings.EMBEDDER == "genai":
        if not _genAIEmbeddingService:
            _genAIEmbeddingService = GenAIEmbeddingService()
        return _genAIEmbeddingService
    elif settings.EMBEDDER == "mistralai":
        if not _mistralAIEmbeddingService:
            _mistralAIEmbeddingService = MistralAIEmbeddingService()
        return MistralAIEmbeddingService()

    raise RuntimeError("unsupported embedder")
