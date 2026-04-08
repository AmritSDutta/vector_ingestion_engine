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

    APP_NAME: str = "InsightScope"
    ENV: str = "dev"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    EMBEDDER: str = "google"
    VECTOR_STORE: str = "chroma"
    CHROMA_DIR: str = "data/chroma"

    EMBEDDING_DIM: int = 256
    EMBEDDING_MODEL: str = "text-embedding-004"
    GENAI_MODEL: str = "gemini-2.5-flash-lite"
    OPENAI_MODEL: str = "gpt-5-nano"

    FILE_STORE_DIR: str = "tmp_ingest"
    CHUNK_SIZE: int = 30
    CHUNK_OVERLAP: int = 5
    PDF_EXTRACTOR_MODEL: str = "gemini-2.5-flash"

    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        logging.info("Settings Loaded")
        print(DOTENV_PATH)
        logging.info(f"CWD: {os.getcwd()}")
        logging.info(f"model_dump: {_settings.model_dump()}")
    return _settings
