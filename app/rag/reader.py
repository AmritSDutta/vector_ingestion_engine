import logging
import os
import shutil
from pathlib import Path

from app.config.config import get_settings


def read_files(files: list[str]) -> Path:
    # Normalize inputs, detect pdf/image/text and return list of {"source":, "text":, "meta":}
    # TODO: call pdf parser / image OCR adapters
    settings = get_settings()
    root_path: Path = Path(__file__).resolve().parent.parent.parent
    logging.info(f'root dir: {root_path}')
    logging.info(f'to be stored in from settings: {settings.FILE_STORE_DIR}')
    logging.info(f'to be stored in: tmp_ingest')
    dest_dir = root_path / 'tmp_ingest'
    logging.info(f'dest dir: {dest_dir}')
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        src = Path(f)
        with open(src, "rb") as inp, open(dest_dir / src.name, "wb") as out:
            out.write(inp.read())
    return dest_dir
