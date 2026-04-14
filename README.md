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
*   **Optimized PG Schema:** Native Full-Text Search (`tsvector`) and high-performance array searching (`TEXT[]`) with `asyncpg` connection pooling and double-checked locking.

### 4. Enterprise-Grade Security & Eval
*   **HTTP Basic Auth:** Secure access to all `/api` endpoints with timing-attack resistant digest comparison and a configurable toggle.
*   **PII Redaction:** Automatic anonymization of sensitive data using **Microsoft Presidio**.
*   **Content Moderation:** Integrated with **OpenAI Moderation API** and custom pattern-based threat detection.
*   **RAG Evaluation:** Automated faithfulness and context relevancy scoring using **LlamaIndex** and OpenAI `gpt-4o-mini`.
*   **Rate Limiting:** Protects endpoints using `fastapi-limiter`.

---

## 🔒 Security & Authentication

The engine implements **HTTP Basic Authentication** for all endpoints under the `/api` prefix. This layer ensures that only authorized clients can trigger ingestion or perform queries.

### Key Features:
- **Digest Comparison:** Uses `secrets.compare_digest` to prevent timing attacks during credential verification.
- **Configurable Toggle:** Authentication can be enabled or disabled via the `IS_AUTH_ENABLED` flag.
- **Global Protection:** Enforced at the router level in `app/main.py`.

### Configuration:
To configure authentication, add the following to your `.env` file:
```env
IS_AUTH_ENABLED=True
API_USERNAME=admin
API_PASSWORD=your_secure_password
```

---

## 💎 Mission-Critical Endpoint: Secure Ingestion

The `/upload_to_all_database` endpoint serves as the **Secure Command Center** of the ingestion pipeline. It orchestrates a complex, non-blocking workflow designed for high-concurrency environments:

### 🛠️ Technical Workflow & Tech Stack
*   **Restricted Entry (SlowAPI):** Protected by a strict rate limiter (1 request per 30 minutes) to ensure system stability.
*   **Async Orchestration (Celery + Redis):** Offloads heavy processing to background workers, returning a `task_id` for real-time status tracking.
*   **Privacy-First "Dark Processing" (Microsoft Presidio):** Deep entity recognition and anonymization scrub the text of all sensitive identity markers before they ever reach an LLM or database.
*   **Neural Transformation (Gemini/Mistral AI):** Sanitized data is converted into high-dimensional embeddings using a factory-patterned AI service.
*   **Atomic Multi-Sync (Triple-Write):** Simultaneously synchronizes data across three distinct vector architectures: **Qdrant**, **Milvus**, and **PostgreSQL (pgvector)**.

### 🎨 Visual Synthesis (Image Generation Prompt)
> "A master-control room for a global AI network. In the center, a holographic interface shows a file being uploaded to '/upload_to_all_database' with a 'SECURE' status bar. To the left, a digital 'X-Ray' scanner (Microsoft Presidio) is highlighting and redacting names and dates from scrolling text. In the center-top, a brain-like neural lattice (Gemini/Mistral AI) is glowing as it turns the text into golden geometric vectors. On the right, three massive, distinct server monoliths labeled 'QDRANT', 'MILVUS', and 'POSTGRES' are receiving synchronized streams of golden energy. Below the interface, a 'CELERY' status dashboard shows active task progress bars moving in real-time. Hyper-realistic, 8k resolution, cinematic lighting with a mix of deep-sea blue and electric gold accents."

---

## 🎯 Architecture Overview

This system implements a **CQRS-style separation** between write and read operations:

```
┌─────────────────────────────────────────────────────────┐
│                   INGESTER APP                          │
│  (Write Side - Separate Application Logic)               │
│  • create()              - Create collections              │
│  • save()                - Batch insert documents         │
│  • delete_collection()   - Drop collections              │
│  • Data migration, ETL, schema management              │
└───────────────┬─────────────────────────────────────────┘
                │
                │ (Shared Vector Database)
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│              SEARCH API (This App)                       │
│  (Read Side - Query Optimized)                            │
│  • query()               - Dense vector search            │
│  • hybrid_search()       - Multi-stage hybrid search      │
│  • list_collection()     - Metadata queries               │
│  • Connection pooling, caching, read replicas           │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ System Architecture

```mermaid
graph TD
    subgraph Frontend [Streamlit UI]
        UI[app/ui.py]
    end

    subgraph Backend [FastAPI Application]
        API[app/main.py]
        RV[request_validator.py]
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
            PR[PII_Redactor.py]
        end
        
        subgraph RAG_Utilities
            EV[app/rag/eval.py]
            RD[app/rag/reader.py]
            TH[time_helper.py]
        end
    end

    subgraph AsyncTasks [Task Queue]
        CT[app/celery_task.py]
        CW[app/celery_worker.py]
        RD_REDIS[(Redis)]
    end

    subgraph "Vector Stores (Pluggable)"
        QD[(Qdrant)]
        MV[(Milvus)]
        PG[(Postgres/pgvector)]
    end

    UI <--> API
    API --> RV
    RV --> Router
    Router --> IR & QR
    IR -- Trigger --> CT
    CT --> CW
    CW <--> RD_REDIS
    CW --> IS
    QR --> QS
    QS --> EF & VF
    
    QS -.-> EV
    QS -.-> TH
    
    VF --> QD | settings.VECTOR_STORE='qdrant' | QD
    VF --> MV | settings.VECTOR_STORE='milvus' | MV
    VF --> PG | settings.VECTOR_STORE='postgres' | PG
```

---

## 🚀 Execution Guide: Step-by-Step

Follow these steps in the exact order to launch the full distributed system.

### 1. Prerequisites & Environment
*   **Python:** 3.12+ (Recommended)
*   **Redis:** Required for Celery.
*   **Vector DB:** An instance of Qdrant, Milvus, or Postgres (with pgvector).

**Setup `.env`:**
Create a `.env` file in the root directory:
```env
# API Keys
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key

# Authentication (Optional/Toggleable)
IS_AUTH_ENABLED=True
API_USERNAME=admin
API_PASSWORD=your_secure_password

# Infrastructure
REDIS_URL=redis://localhost:6379/0
VECTOR_STORE=milvus  # Options: qdrant, milvus, postgres

# Vector DB Config
MILVUS_URI=localhost
MILVUS_TOKEN=your_token
QDRANT_HOST=localhost
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
*   **`app/celery_task.py`**: Task definitions and progress callback management.
*   **`app/celery_worker.py`**: Singleton Celery instance and worker config.
*   **`app/rag/`**: RAG evaluation logic (`eval.py`) and file reading utilities (`reader.py`).
*   **`app/services/`**: Core logic layer.
    *   **`ingest_service.py`**: Handles loading, PII redaction, and multi-store distribution.
    *   **`query_service.py`**: Orchestrates retrieval with latency tracking.
    *   **`vector_store/`**: Factory for Qdrant, Milvus, and Postgres.
*   **`app/routers/`**: REST API endpoints with rate limiting and `request_validator.py`.
*   **`app/ui.py`**: Streamlit-based InsightScope dashboard.

---

## 🧪 Testing & Validation
Run the comprehensive test suite and static type checks:
```bash
# Run Unit Tests
pytest

# Run Type Checking
pip install mypy
mypy app
```

## 🛤️ Future Enhancements
- [ ] **Apache Airflow Integration:** Multi-stage ETL for high-volume document pipelines.
- [ ] **OCR/PDF Adapter:** Native PDF extraction with layout analysis.
- [ ] **Complex Filtering:** Skill-based matching logic (e.g., "years of experience" extraction).
