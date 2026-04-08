import json
import logging
from typing import List, Dict, Any

from chromadb import QueryResult

from app.config.factory import get_vector_store, get_embedding_service
from app.rag.eval import evaluate_answer
from app.rag.retrieval import rewrite_query_llm, retrieve
from app.rag.synthesiser import get_synthesiser

MAX_CHARS = 4000
PII_HINTS = ["OCR SAMPLE BLOCK", "Device Calibration Code", "Calibration Code", "XJ29-QR7"]


async def query_handler(user_query: str, top_k: int = 20):
    emb = get_embedding_service()
    store = get_vector_store()

    hits: QueryResult = retrieve(user_query.strip(), emb, store, top_k=top_k)
    logging.info(f'RAG hits: \n{json.dumps(hits, indent=2)}')
    rewritten: str = rewrite_query_llm(user_query)
    contexts = result_json_to_contexts(hits, include_meta_keys=["page", "title", "url", 'source', 'doc', 'text'])

    synthesized_answer = get_synthesiser(rewritten if rewritten and len(rewritten) > 10 else user_query, contexts)
    evaluation = evaluate_answer(user_query, synthesized_answer, contexts)
    return {
        "query": user_query,
        "rewritten": rewritten,
        "answer": synthesized_answer,
        'evaluation': evaluation
    }


def _truncate(s: str, max_chars=MAX_CHARS) -> str:
    return s if not s or len(s) <= max_chars else s[:max_chars - 3] + "..."


def chroma_hits_to_contexts(hits: dict, include_meta_keys: List[str] = None) -> List[str]:
    include_meta_keys = include_meta_keys or []
    ids = (hits.get("ids") or [[]])[0]  # QueryResult often nests lists
    docs = (hits.get("documents") or [[]])[0]
    metas = (hits.get("metadatas") or [[]])[0]
    dists = (hits.get("distances") or [[]])[0]

    contexts = []
    n = max(len(ids), len(docs), len(metas), len(dists))
    for i in range(n):
        src = ids[i] if i < len(ids) else f"chunk-{i}"
        # sometimes docs contain JSON fragments or lists; coerce to str then compact
        raw_doc = docs[i] if i < len(docs) else ""
        if isinstance(raw_doc, (list, dict)):
            try:
                raw_doc = json.dumps(raw_doc, ensure_ascii=False)
            except Exception:
                raw_doc = str(raw_doc)
        raw_doc = str(raw_doc).replace("\r", " ").replace("\n\n", "\n").strip()
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None

        meta_parts = [f"source_id:{src}"]
        for k in include_meta_keys:
            v = meta.get(k)
            if v:
                meta_parts.append(f"{k}:{v}")
        if dist is not None:
            meta_parts.append(f"score:{dist:.4f}")

        header = "[DOCUMENT] " + " | ".join(meta_parts) + "\n"
        body = _truncate(raw_doc)
        # add small PII hint if suspicious tokens found (helps the safety classifier)
        hints = [h for h in PII_HINTS if h in body]
        if hints:
            body = f"[POSSIBLE_OCR/PII_HINTS: {', '.join(hints)}]\n" + body

        contexts.append(header + body)
    return contexts


def result_json_to_contexts(data: Dict[str, Any],
                            include_meta_keys: List[str] = None) -> List[str]:
    include_meta_keys = include_meta_keys or []
    results = data.get("results", [])
    contexts = []

    for idx, item in enumerate(results):
        payload = item.get("payload") or item
        text = payload.get("doc") or payload.get("text") or ""
        meta = {k: v for k, v in payload.items() if k not in ("doc", "text")}

        # normalize text
        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False)
        text = str(text).replace("\r", " ").replace("\n\n", "\n").strip()

        # metadata section
        meta_parts = [f"result_idx:{idx}"]
        for k in include_meta_keys:
            v = meta.get(k)
            if v is not None:
                meta_parts.append(f"{k}:{v}")

        header = "[DOCUMENT] " + " | ".join(meta_parts) + "\n"
        contexts.append(header + text)

    return contexts
