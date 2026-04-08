import logging
from typing import List, Optional

from google import genai
from google.genai import types, Client
from pydantic import BaseModel, Field

from app.config.config import Settings

_client: Client | None = None


#  Define the Pydantic Schema ---
class TableData(BaseModel):
    markdown: str = Field(..., description="The extracted table converted to Markdown format.")


class FigureData(BaseModel):
    caption: str = Field(..., description="A detailed description or caption of the image, chart, or diagram.")


class PageContent(BaseModel):
    page_no: int = Field(..., description="The page number.")
    text: str = Field(..., description="The main text content extracted from the page.")
    tables: List[TableData] = Field(default_factory=list, description="List of tables found on this page.")
    figures: List[FigureData] = Field(default_factory=list, description="List of figures/images found on this page.")


class PDFExtraction(BaseModel):
    pages: List[PageContent]


def get_genai_client():
    global _client
    if not _client:
        _client = genai.Client()
    return _client


def extract_through_llm(pdf_path: str):
    """
    extract table, image, jsons objs present in PDF.
    :param pdf_path:
    :return:
    """
    settings = Settings()
    # 3. Load your PDF
    # For files < 20MB, you can pass bytes directly. For larger files, use the Files API.
    # pdf_path = "your_document.pdf"
    logging.info(f'pdf path: {pdf_path}')
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    prompt = """
    Process this PDF page by page. For each page:
    1. Identify all tables and convert them to valid Markdown strings with high accuracy.
    2. Identify all images, charts, or diagrams and provide a detailed caption/description for them.
    3. Identify all json data if present, and convert them to valid markdown with high accuracy.
    
    Ensure the output strictly follows the JSON schema provided.
    """

    logging.info("Processing PDF...")
    response = get_genai_client().models.generate_content(
        model=settings.PDF_EXTRACTOR_MODEL,
        contents=[
            types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PDFExtraction,  # Pass the Pydantic class here
            # temperature=0.1  # Low temperature for factual extraction
        )
    )

    # The response is already valid JSON because of the schema
    try:
        logging.info("Processing PDF complete")
        extraction_result = response.parsed

        # Iterate through the structured data
        for page in extraction_result.pages:
            logging.info(f"\n=== Page {page.page_no} ===")

            # 1. Print Text snippet
            logging.info(f"\n[Text Snippet]: {page.text[:100]}...")

            # 2. Print Tables
            if page.tables:
                print(f"\n[Tables Found: {len(page.tables)}]")
                for t in page.tables:
                    logging.info("--- Markdown Table ---")
                    logging.info(t.markdown)

            # 3. Print Figures
            if page.figures:
                logging.info(f"\n[Figures Found: {len(page.figures)}]")
                for f in page.figures:
                    logging.info(f"- Caption: {f.caption}")

        return extraction_result

    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        logging.warning("Raw Output:", response.text)
