import logging
from pathlib import Path

from app.rag.index_builder import build_index, delete_index, list_chroma_collections
from app.rag.reader import read_files


def ingest_and_store_embedding(files, fileName: str):
    logging.info(f'uploading files stored in : {files}')
    files_path: Path = read_files(files)
    logging.info(f'uploaded files stored in : {files_path.name}')
    build_index(fileName)
    logging.info(f'indexes built and stored in vector store')
    return files


def delete_store_embedding(name_of_collection: str):
    logging.info(f'deleting index collection : {name_of_collection}')
    delete_index(name_of_collection)
    logging.info(f'deleted index collection : {name_of_collection}')


def list_collections_chroma() -> list[str]:
    logging.info(f'listing index collection')
    return list_chroma_collections()
