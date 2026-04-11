# 🚀 Vector Ingestion Engine

A high-performance, production-ready Retrieval-Augmented Generation (RAG) pipeline designed for complex document ingestion, semantic retrieval, and multi-stage search orchestration. This engine is specialized for processing resumes and leverages advanced embedding techniques, hybrid search (Dense + Sparse), and cross-encoder reranking.

---

## 🌟 Key Features

### 1. Asynchronous Ingestion Pipeline
*   **Non-Blocking Workflow:** Uses **Celery** and **Redis** to handle heavy document processing, chunking, and embedding generation in the background.
*   **Status Polling:** Real-time progress updates (Loading -> Embedding -> Saving) via a background poller in the FastAPI console.

### 2. Advanced Retrieval Strategies
*   **Hybrid Search:** Combines Dense vectors (Gemini/Mistral) with Sparse vectors (BM25) using **Reciprocal Rank Fusion (RRF)** for superior accuracy.
*   **Late Interaction:** Support for ColBERT vectors via FastEmbed.
*   **Cross-Encoder Reranking:** Precision re-scoring using `jina-reranker-v2-base-multilingual`.

### 3. Pluggable Vector Infrastructure
*   **Multi-Store Support:** Seamlessly switch between **Qdrant**, **Milvus**, and **PostgreSQL (pgvector)** via the Factory pattern.
*   **Optimized PG Schema:** Native Full-Text Search (`tsvector`) and high-performance array searching (`TEXT[]`) for skills matching.

### 4. Enterprise-Grade Security & Eval
*   **Rate Limiting:** Protects endpoints using `fastapi-limiter`.
*   **RAG Evaluation:** Automated faithfulness and context relevancy scoring using OpenAI `gpt-4o-mini`.
*   **Input Sanitization:** Pattern-based threat detection for all user queries.

---

## 🛠️ System Architecture

```mermaid
graph TD
    subgraph Frontend [Streamlit UI]
        UI[app/ui.py]
    end

    subgraph Backend [FastAPI Application]
        API[app/main.py]
        Router[app/routers/app_router.py]
        
        subgraph Routers
            IR[routes_ingest.py]
            QR[routes_query.py]
        end
        
        subgraph Services
            IS[ingest_service.py]
            QS[query_service.py]
            EF[EmbeddingFactory.py]
            VF[VectorStoreFactory.py]
        end
    end

    subgraph AsyncTasks [Task Queue]
        CW[app/celery_worker.py]
        RD[(Redis)]
    end

    subgraph "Vector Stores (Pluggable)"
        QD[(Qdrant)]
        MV[(Milvus)]
        PG[(Postgres/pgvector)]
    end

    UI <--> API
    API --> Router
    Router --> IR & QR
    IR -- Trigger --> CW
    CW <--> RD
    CW --> IS
    QR --> QS
    
    IS --> EF & VF
    QS --> EF & VF
    
    VF --> QD | settings.VECTOR_STORE='qdrant' | QD
    VF --> MV | settings.VECTOR_STORE='milvus' | MV
    VF --> PG | settings.VECTOR_STORE='postgres' | PG
```

---

## 🚀 Execution Guide: Step-by-Step

Follow these steps in the exact order to launch the full distributed system.

### 1. Prerequisites & Environment
*   **Python:** 3.10+
*   **Redis:** Required for Celery.
*   **Vector DB:** An instance of Qdrant, Milvus, or Postgres (with pgvector).

**Setup `.env`:**
Create a `.env` file in the root directory:
```env
# API Keys
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key

# Infrastructure
REDIS_URL=redis://localhost:6379/0
VECTOR_STORE=qdrant  # Options: qdrant, milvus, postgres

# Vector DB Config
QDRANT_HOST=localhost
QDRANT_PORT=6333
DB_DSN=postgres://user:pass@localhost/resume_vector_db
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Start the Distributed Components

#### Step A: Start Redis (The Message Broker)
```bash
# Recommended: Using Docker
docker run -d -p 6379:6379 redis
```

#### Step B: Start Celery Worker (The Task Processor)
Open a new terminal and run:
*   **Windows (Required):**
    ```bash
    celery -A app.celery_worker worker --loglevel=info -P solo
    ```
*   **Linux/Mac:**
    ```bash
    celery -A app.celery_worker worker --loglevel=info
    ```

#### Step C: Start FastAPI Backend (The API)
Open a new terminal and run:
```bash
python app/main.py
```

#### Step D: Start Streamlit Frontend (The UI)
Open a new terminal and run:
```bash
streamlit run app/ui.py
```

---

## 📂 Project Structure

*   **`app/main.py`**: Entry point for the FastAPI application.
*   **`app/celery_worker.py`**: Singleton Celery instance and worker config.
*   **`app/celery_task.py`**: Celery task wrappers for background execution.
*   **`app/services/`**: Core logic (Ingestion, Querying, Embedding/Vector Factories).
*   **`app/routers/`**: REST API endpoints with rate limiting and background polling.
*   **`app/ui.py`**: Streamlit-based InsightScope dashboard.

---

## 📝 SQL Schema (For PostgreSQL/pgvector)

If using the `postgres` vector store, execute the following to set up your schema:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS resume_details (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    resume_id TEXT UNIQUE,
    name TEXT,
    category TEXT,
    skills TEXT[],
    overall TEXT,
    embedding VECTOR(1024), -- Dimension for Gemini
    fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(overall, ''))) STORED
);

CREATE INDEX IF NOT EXISTS idx_resume_hnsw ON resume_details USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_resume_fts ON resume_details USING gin (fts_vector);
CREATE INDEX IF NOT EXISTS idx_resume_skills ON resume_details USING gin (skills);
```

---

## 🧪 Testing
Run the comprehensive test suite to verify ingestion and retrieval:
```bash
pytest
```

## 🛤️ Future Enhancements
- [ ] **Apache Airflow Integration:** Multi-store distribution and embedding caching (Plan in `AIRFLOW_MULTI_STORE_IMPLEMENTATION_PLAN.md`).
- [ ] **Advanced OCR:** Integration for complex PDF parsing.
- [ ] **OAuth2 Integration:** Secure API access.
