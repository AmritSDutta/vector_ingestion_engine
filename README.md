# 🚀 Vector Ingestion Engine

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for processing, storing, and querying document data (specialized for resumes) using advanced embedding techniques, hybrid search, and reranking.

## 🌟 Overview

The **Vector Ingestion Engine** provides a full-featured RAG pipeline. It ingests documents, generates high-dimensional embeddings, stores them in specialized vector databases (Qdrant, Milvus, or PostgreSQL), and performs semantic searches enhanced by reranking and multi-stage retrieval strategies.

### Key Features
- **Multi-Vector Support:** Implements dense (Gemini/Mistral), sparse (BM25), and late interaction (ColBERT) vector storage.
- **Hybrid Search:** 
  - **Qdrant:** Advanced multi-stage search (Dense + Sparse -> RRF -> ColBERT Rerank) in a single call.
  - **Milvus:** RRF-based hybrid search combining dense and BM25 vectors.
- **Cross-Encoder Reranking:** Uses `jina-reranker-v2-base-multilingual` for precision re-scoring.
- **Flexible Vector Stores:** Pluggable support for **Qdrant**, **Milvus**, and **PostgreSQL (pgvector)**.
- **RAG Evaluation:** Automated evaluation of faithfulness and context relevancy using LlamaIndex and OpenAI.
- **Security First:** Robust request validation with pattern-based threat detection and OpenAI moderation.
- **Modern Stack:** Built with FastAPI (Backend) and Streamlit (Frontend).

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
            MR[matching_doc_router.py]
        end
        
        subgraph Services
            IS[ingest_service.py]
            QS[query_service.py]
            EF[EmbeddingFactory.py]
            VF[VectorStoreFactory.py]
        end
    end

    subgraph "Vector Stores (Pluggable)"
        QD[(Qdrant)]
        MV[(Milvus)]
        PG[(Postgres/pgvector)]
    end

    UI <--> API
    API --> Router
    Router --> IR & QR & MR
    IR --> IS
    QR --> QS
    MR --> QS
    
    IS --> EF & VF
    QS --> EF & VF
    
    VF --> QD | settings.VECTOR_STORE='qdrant' | QD
    VF --> MV | settings.VECTOR_STORE='milvus' | MV
    VF --> PG | settings.VECTOR_STORE='postgres' | PG
```

### 📂 Project Structure Overview

The project is organized into a clean, service-oriented architecture:

*   **`app/ui.py`**: The **InsightScope** dashboard for file uploads, collection management, and semantic querying.
*   **`app/services/`**: The core logic layer.
    *   **`ingest_service.py`**: Handles document loading, text chunking, and embedding generation.
    *   **`query_service.py`**: Orchestrates multi-stage retrieval (Dense + Sparse + Reranking).
    *   **`vector_store/`**: Implements the Factory pattern to switch between Qdrant, Milvus, and Postgres.
*   **`app/routers/`**: Defines the RESTful API contract for the frontend and external clients.
*   **`app/config/`**: Centralized configuration using Pydantic `Settings` for environment variable management.

### 🚀 Data Flow (Ingestion & Query)

1.  **Ingestion**: `Document -> Chunking -> Embedding (Gemini/Mistral) -> Vector Store (Named Vectors)`
2.  **Query**: `User Query -> Embedding -> Vector Search -> Hybrid Fusion (RRF) -> Reranking (Jina) -> LLM Answer`

### SQL
```
-- 1. Setup Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Schema Definition
CREATE TABLE IF NOT EXISTS resume_details (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    resume_id TEXT UNIQUE,
    name TEXT,
    category TEXT,
    education TEXT,
    skills TEXT[],
    summary TEXT,
    overall TEXT,
    embedding VECTOR(1536), -- Dimension should match your embedding model
    fts_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(overall, ''))
    ) STORED
);

-- 3. Indexing
-- HNSW index for high-performance vector similarity search
CREATE INDEX IF NOT EXISTS idx_resume_details_embedding_hnsw 
ON resume_details USING hnsw (embedding vector_cosine_ops);

-- GIN indexes for Full Text Search and Array lookups
CREATE INDEX IF NOT EXISTS idx_resume_details_fts_gin ON resume_details USING gin (fts_vector);
CREATE INDEX IF NOT EXISTS idx_resume_details_skills_gin ON resume_details USING gin (skills);

-- 4. Operations (Reference Implementation)

-- Upsert: Inserts or updates existing resume records
-- Parameters: $1: resume_id, $2: name, $3: category, $4: education, $5: skills, $6: summary, $7: overall, $8: embedding
INSERT INTO resume_details 
(resume_id, name, category, education, skills, summary, overall, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
ON CONFLICT (resume_id) DO UPDATE SET
    embedding = EXCLUDED.embedding,
    overall = EXCLUDED.overall,
    summary = EXCLUDED.summary,
    skills = EXCLUDED.skills;

-- Semantic Search: Finds top-K results by cosine distance
-- Parameters: $1: query_embedding, $2: limit
SELECT resume_id, name, category, education, skills, summary, overall,
       (embedding <=> $1::vector) as distance
FROM resume_details
ORDER BY distance ASC
LIMIT $2;

-- Hybrid Search: Combines Vector Search and Full Text Search using RRF
-- Parameters: $1: query_embedding, $2: limit, $3: text_query
WITH vector_search AS (
    SELECT resume_id, row_number() OVER (ORDER BY embedding <=> $1::vector) as rank
    FROM resume_details
    ORDER BY embedding <=> $1::vector
    LIMIT $2 * 4
),
text_search AS (
    SELECT resume_id, row_number() OVER (ORDER BY ts_rank(fts_vector, websearch_to_tsquery('english', $3)) DESC) as rank
    FROM resume_details
    WHERE fts_vector @@ websearch_to_tsquery('english', $3)
    LIMIT $2 * 4
)
SELECT 
    r.id, r.resume_id, r.name, r.category, r.education, r.skills, r.summary,
    (COALESCE(1.0 / (60 + vs.rank), 0.0) + COALESCE(1.0 / (60 + ts.rank), 0.0)) as rrf_score
FROM vector_search vs
FULL OUTER JOIN text_search ts ON vs.resume_id = ts.resume_id
JOIN resume_details r ON r.resume_id = COALESCE(vs.resume_id, ts.resume_id)
ORDER BY rrf_score DESC
LIMIT $2;
```

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
