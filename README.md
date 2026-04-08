# 🚀 Vector Ingestion Engine

A Retrieval-Augmented Generation (RAG) application for processing, storing, and querying document data using Gemini and Qdrant.

## 🌟 Overview

The **Vector Ingestion Engine** provides a full pipeline to ingest documents (resumes), generate embeddings using Google's Gemini API, store them in a Qdrant vector database, and perform semantic searches with reranking.

### Key Features
- **FastAPI Backend:** High-performance API for ingestion and querying.
- **Streamlit Frontend:** User-friendly interface for document upload and search.
- **Gemini AI:** Leveraging Google's Generative AI for high-quality embeddings.
- **Qdrant Vector Store:** Efficient vector storage and similarity search.
- **Advanced Retrieval:** Includes a reranking step using Cross-Encoders for improved accuracy.

## 🛠️ Architecture

- **`app/main.py`**: FastAPI entry point.
- **`app/ui.py`**: Streamlit UI.
- **`app/services/`**: Core business logic.
    - `ingest_service.py`: Handles document processing and embedding.
    - `query_service.py`: Manages the retrieval and reranking pipeline.
    - `embedding/`: Factory for different embedding providers (GenAI, Mistral).
    - `vector_store/`: Abstractions for Qdrant and other vector databases.
- **`app/routers/`**: API endpoint definitions.
- **`app/config/`**: Pydantic-based configuration and logging.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Qdrant](https://qdrant.tech/documentation/guides/installation/) (Running locally on port 6333 or via Cloud)
- Google API Key (for Gemini)

### Installation
1. Clone the repository.
2. Create a `.env` file in the root:
   ```env
   GOOGLE_API_KEY=your_key_here
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### 1. Start the Backend
```bash
python -m uvicorn app.main:app --reload
```

#### 2. Start the Frontend
```bash
streamlit run app/ui.py
```

### Testing
```bash
pytest
```

## 📝 Current Limitations & Roadmap
- [ ] Implement multi-file upload processing logic in `routes_ingest.py`.
- [ ] Complete `delete` and `list_collections` endpoints in the router.
- [ ] Fix typo in `app/main.py` (`" __main__"` vs `"__main__"`).
- [ ] Integrate `vector_store.save()` call into `ingest_and_store_embedding` service.
