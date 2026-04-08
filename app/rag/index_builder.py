import json
import logging
import time
import uuid
from pathlib import Path

import pdfplumber
from llama_index.core.node_parser import SentenceSplitter

from .llm_extractor import extract_through_llm
from ..config.config import Settings
from ..config.factory import get_embedding_service, get_vector_store


# LlamaIndex splitter import local to avoid heavy import unless used
def chunk_text_llama(text, chunk_size: int = 600, chunk_overlap: int = 150):
    logging.info(f'chunking text: {text if text else None} . . ,chunk_size: {chunk_size},overlap: {chunk_overlap}')
    if not text:
        logging.warning(f'blank text received')
        return

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.split_text(text)
    return [getattr(n, "get_content", lambda: str(n))() for n in nodes]


def extract_text_from_pdf(path: Path) -> str:
    logging.info(f'extracting text from: {path}')
    txt = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt += (p.extract_text() or "") + "\n"
    return txt


def extract_text(src: Path) -> (str, str):
    """Return mapping filename -> text for PDFs in src directory."""
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"source dir missing: {src}")
    result = {}
    file_name = f'recent_file_{time.time()}'
    for p in src.iterdir():
        if not p.is_file():
            continue
        file_name = p.name
        if p.suffix.lower() != ".pdf":
            logging.info("skipping non-pdf: %s", p.name)
            continue
        result[p.name] = extract_text_from_pdf(p)
        try:
            extraction_result = extract_through_llm(src.name + '/' + p.name)
            result[p.name] = result[p.name] + extraction_result.model_dump_json()
        except Exception as e:
            logging.error(e)

    full_text = "\n\n".join(result.values())
    return full_text if full_text else '', file_name


def build_index(fileName: str =''):
    # 1) chunk docs (call chunker)
    settings = Settings()
    root_path: Path = Path(__file__).resolve().parent.parent.parent
    src = root_path / Path(settings.FILE_STORE_DIR)

    out = Path("data/chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    text, file_name = extract_text(src)
    chunks = chunk_text_llama(text,
                              settings.CHUNK_SIZE,
                              settings.CHUNK_OVERLAP)
    with out.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(chunks):
            fh.write(json.dumps({"id": f"chunk-{i}", "text": c, "source": fileName}) + "\n")
    logging.info(f"wrote {len(chunks)} chunks -> {out}, chunk size: {settings.CHUNK_SIZE},"
                 f" chunk overlap: {settings.CHUNK_OVERLAP}")

    # extract_llm(src.name)

    # 2) embed batches
    embedder = get_embedding_service()
    coll_name: str = fileName.strip() if fileName.strip() else uuid.uuid4().hex
    store = get_vector_store()
    store.create()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    metas = [{"source": row["source"], "i": i} for i, row in enumerate(rows)]
    embs = embedder.embed(texts)
    logging.info(f"indexed {len(ids)} chunks")

    # 3) upsert to vector store with metadata
    store.save(ids, texts, metas, embs)


def delete_index(collection_name: str):
    store = get_vector_store()
    store.delete_collection(collection_name)


def list_chroma_collections() -> list[str]:
    store = get_vector_store()
    return store.list_collection()
