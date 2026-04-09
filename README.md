# 🚀 Vector Ingestion Engine

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for processing, storing, and querying document data (specialized for resumes) using advanced embedding techniques, hybrid search, and reranking.

## 🌟 Overview

The **Vector Ingestion Engine** provides a full-featured RAG pipeline. It ingests documents, generates high-dimensional embeddings, stores them in specialized vector databases (Qdrant or Milvus), and performs semantic searches enhanced by reranking and hybrid retrieval strategies.

### Key Features
- **Multi-Vector Support:** Implements dense, sparse (BM25), and late interaction (ColBERT) vector storage.
- **Hybrid Search & Reranking:** Combines vector similarity with BM25 and uses Cross-Encoders (Jina AI) for precise reranking.
- **Flexible Vector Stores:** Pluggable support for **Qdrant** and **Milvus**.
- **Embedding Providers:** Integrated with **Google Gemini** and **Mistral AI**.
- **RAG Evaluation:** Automated evaluation of faithfulness and context relevancy using LlamaIndex and OpenAI.
- **Security First:** Robust request validation with pattern-based threat detection and OpenAI moderation.
- **Modern Stack:** Built with FastAPI (Backend) and Streamlit (Frontend).

## 🛠️ Architecture

- **`app/main.py`**: FastAPI entry point and application lifecycle management.
- **`app/ui.py`**: Streamlit-based user interface for ingestion and querying.
- **`app/services/`**: Core business logic layer.
    - `ingest_service.py`: Orchestrates document loading, embedding, and storage.
    - `query_service.py`: Manages the retrieval pipeline, including reranking.
    - `embedding/`: Factory and implementations for Gemini and Mistral AI embeddings.
    - `vector_store/`: Abstractions and implementations for Qdrant and Milvus.
- **`app/routers/`**: FastAPI routers for ingestion, querying, and document matching.
- **`app/rag/`**: RAG-specific utilities.
    - `reader.py`: Placeholder for advanced document parsing (PDF/OCR).
    - `eval.py`: Evaluation suite using LlamaIndex evaluators.
- **`app/config/`**: Centralized Pydantic-based configuration and custom logging.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- **Qdrant** (port 6333) or **Milvus** instance.
- API Keys: `GOOGLE_API_KEY` (Gemini), `MISTRAL_API_KEY` (optional), `OPENAI_API_KEY` (for evaluation & moderation).

### Installation
1. Clone the repository.
2. Create a `.env` file in the root (see `app/config/config.py` for all options):
   ```env
   GOOGLE_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   VECTOR_STORE_TYPE=qdrant
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### 1. Start the Backend (FastAPI)
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Note: Ensure the typo in `app/main.py` is fixed for direct execution.*

#### 2. Start the Frontend (Streamlit)
```bash
streamlit run app/ui.py
```

### Testing
```bash
pytest
```

## 📝 Current Roadmap & TODOs
- [ ] Fix typo in `app/main.py` (`" __main__"` -> `"__main__"`).
- [ ] Implement `QdrantStore.hybrid_search()`.
- [ ] Complete `list_collections` endpoint in `routes_ingest.py`.
- [ ] Integrate advanced PDF/OCR parsing in `app/rag/reader.py`.
- [ ] Expand hybrid search capabilities in the UI.
