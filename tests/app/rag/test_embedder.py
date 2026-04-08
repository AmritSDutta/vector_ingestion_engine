import pytest

from app.rag.embedder import GoogleEmbedder


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_google_embedder_test():
    embedder = GoogleEmbedder()
    to_be_embedded = 'please embed me'
    embeddings: list[list[float]] = embedder.embed([to_be_embedded])
    assert len([to_be_embedded]) == len(embeddings)
