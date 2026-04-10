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