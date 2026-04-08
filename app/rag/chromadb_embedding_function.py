from chromadb import EmbeddingFunction, Documents, Embeddings
from google.genai import client, types


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    can be used as embedding function by chroma
    """
    def __call__(self, input: Documents) -> Embeddings:
        EMBEDDING_MODEL_ID = "text-embedding-004"
        title = "Custom query"
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_ID,
            contents=input,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                title=title
            )
        )

        return response.embeddings[0].values
