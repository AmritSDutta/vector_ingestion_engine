# 🚀 Vector Ingestion Engine

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for processing, storing, and querying document data (specialized for resumes) using advanced embedding techniques, hybrid search, and reranking.

## 🌟 Overview

The **Vector Ingestion Engine** provides a full-featured RAG pipeline. It ingests documents, generates high-dimensional embeddings, stores them in specialized vector databases (Qdrant or Milvus), and performs semantic searches enhanced by reranking and multi-stage retrieval strategies.

### Key Features
- **Multi-Vector Support:** Implements dense (Gemini/Mistral), sparse (BM25), and late interaction (ColBERT) vector storage.
- **Hybrid Search:** 
  - **Qdrant:** Advanced multi-stage search (Dense + Sparse -> RRF -> ColBERT Rerank) in a single call.
  - **Milvus:** RRF-based hybrid search combining dense and BM25 vectors.
- **Cross-Encoder Reranking:** Uses `jina-reranker-v2-base-multilingual` for precision re-scoring.
- **Flexible Vector Stores:** Pluggable support for **Qdrant** and **Milvus**.
- **RAG Evaluation:** Automated evaluation of faithfulness and context relevancy using LlamaIndex and OpenAI.
- **Security First:** Robust request validation with pattern-based threat detection and OpenAI moderation.
- **Modern Stack:** Built with FastAPI (Backend) and Streamlit (Frontend).

## 🛠️ Architecture

- **`app/main.py`**: FastAPI entry point and application lifecycle management.
- **`app/ui.py`**: Streamlit-based user interface for ingestion and querying.
- **`app/services/`**: Core business logic layer.
    - `ingest_service.py`: Orchestrates document loading, embedding, and storage.
    - `query_service.py`: Manages the retrieval pipeline.
    - `vector_store/`: Abstractions and implementations for Qdrant and Milvus.
- **`app/routers/`**: FastAPI routers for ingestion, querying, and document matching.
- **`app/rag/`**: RAG-specific utilities for reading and evaluation.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- **Qdrant** (port 6333) or **Milvus** instance.
- API Keys: `GOOGLE_API_KEY`, `OPENAI_API_KEY` (for evaluation & moderation).

### Installation
1. Clone the repository.
2. Create a `.env` file in the root:
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
python app/main.py
```

#### 2. Start the Frontend (Streamlit)
```bash
streamlit run app/ui.py
```

### Testing
```bash
pytest
```

## 📝 Current Roadmap & TODOs
- [ ] Implement robust multi-file upload processing in `routes_ingest.py`.
- [ ] Integrate advanced PDF/OCR parsing in `app/rag/reader.py`.
- [ ] Add more RAG evaluators (e.g., Answer Relevancy).
