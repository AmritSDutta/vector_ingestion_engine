from typing import Optional, Sequence
from google import genai


class BaseEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class GoogleEmbedder(BaseEmbedder):
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = 'text-embedding-004',
                 dimension: int = 256
                 ):
        self.api_key = api_key
        self.embedding_model = model
        self.embedding_dimension = dimension
        self._client = genai.Client()

    def embed(self, texts: Sequence[str]) -> list[list[float] | None]:
        resp = self._client.models.embed_content(
            model=self.embedding_model,
            contents=list(texts),
            config={
                "task_type": "retrieval_document",
                "output_dimensionality": self.embedding_dimension,
            }
        )
        return [e.values for e in resp.embeddings]
