import logging
from typing import Any

from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI


def get_synthesiser(query: str, hits_ref: Any):
    llm = OpenAI(model="gpt-5-nano")
    logging.info(f'synthesizing: {query}')

    prompt = f"""
    You generate a quick, concise, factual answer to user query: {query}, 
    based only on the provided retrieved context: {hits_ref}.
    
    Rules for answering:
    1. Do NOT generate synonyms, paraphrasing lists, definitions, or expansions unless explicitly requested.
    2. Do NOT add unrelated background explanations.
    3. Use only information that appears in the retrieved context.
    4. Keep the answer focused, short, and directly responsive to the user query.
    
    Return the answer first, followed by the quality object.
    """

    response: str = llm.predict(PromptTemplate(prompt))
    logging.info(f'Synthesized response: {(response or "")[:100]}')
    return response
