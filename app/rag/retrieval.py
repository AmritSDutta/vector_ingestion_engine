import logging

from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

from app.rag.vector_store import VectorStore


def rewrite_query_llm(query: str) -> str:
    llm = OpenAI(model="gpt-5-nano", reasoning_effort="low")

    template = PromptTemplate(
        """
        Rewrite the search query within 25 to 50 words to improve recall while keeping intent intact.
        Dont repeat same questions."
        User asked: {query}
        
        CONSTRAINT:
        output should be with in 50 words.
        """
    )

    logging.info("prompt template: %s", template.template)
    expanded = llm.predict(template, query=query)  # pass template + named var
    logging.info("expanded or rewritten query: %s", expanded[:50] if expanded else '')
    return expanded


def retrieve(user_query: str, embedder, store: VectorStore, top_k: int = 30):
    qvec = embedder.embed([user_query])[0]
    candidates = store.query(qvec, n_results=top_k, query=user_query)
    return candidates
