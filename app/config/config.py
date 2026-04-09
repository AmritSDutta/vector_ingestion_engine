import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    APP_NAME: str = "VectorIngestionEngine"
    ENV: str = "dev"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    EMBEDDER: str = "genai"
    VECTOR_STORE: str = "milvus"
    CHROMA_DIR: str = "data/chroma"

    EMBEDDING_DIM: int = 1024
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GENAI_MODEL: str = "gemini-2.5-flash-lite"
    OPENAI_MODEL: str = "gpt-5-nano"

    FILE_STORE_DIR: str = "tmp_ingest"
    CHUNK_SIZE: int = 30
    CHUNK_OVERLAP: int = 5
    PDF_EXTRACTOR_MODEL: str = "gemini-2.5-flash"

    COLLECTION_NAME: str = 'resume_details'

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = 'key'
    DATA_FILE_PATH: str = 'data/resume_sample_160.jsonl'

    MILVUS_URI: str = "localhost"
    MILVUS_TOKEN: str = "TOKEN"

    DB_DSN: str = 'postgres://some_user:some_password@localhost/resume_vector_db'
    DB_NAME: str = 'resume_vector_db'
    DB_USER: str = 'some_user'
    DB_PASSWORD: str = 'some_password'
    POSTGRES_BATCH_SIZE: int = 100

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    @property
    def data_file_path(self) -> Path:
        """Return absolute, validated path to the CSV."""
        path = self.BASE_DIR / self.DATA_FILE_PATH
        if not path.exists():
            raise FileNotFoundError(f"data file not found at {path}")
        return path


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        logging.info("Settings Loaded")
        # print(DOTENV_PATH)
        logging.info(f"CWD: {os.getcwd()}")
        # logging.info(f"model_dump: {_settings.model_dump()}")
    return _settings
