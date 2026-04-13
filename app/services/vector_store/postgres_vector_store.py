import asyncio
import logging
from typing import Optional, Sequence, Dict, Any

import asyncpg

from pandas import DataFrame
from app.config.config import get_settings
from app.services.vector_store.vector_store import VectorStore, get_reranker_model, validate_collection_name

logger = logging.getLogger(__name__)


class PGVectorStore(VectorStore):
    def __init__(self):
        settings = get_settings()
        self.conn_str = settings.DB_DSN
        self.collection_name = settings.COLLECTION_NAME
        self.reranker = get_reranker_model()
        self.pool = None
        self.pool_lock = asyncio.Lock()

    async def _get_pool(self):
        if self.pool is None:
            async with self.pool_lock:
                if self.pool is None:
                    self.pool = await asyncpg.create_pool(self.conn_str, min_size=3, max_size=10)
        return self.pool

    async def create(self):
        """Initializes the database schema if it doesn't exist."""
        coll_name = self.collection_name
        validate_collection_name(coll_name)
        logger.info(f"Initializing Postgres collection: {coll_name}")

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                # Enable extensions
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create Table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {coll_name} (
                        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        resume_id TEXT UNIQUE,
                        name TEXT,
                        category TEXT,
                        education TEXT,
                        skills TEXT[],
                        summary TEXT,
                        phone TEXT NULL,
                        location TEXT NULL,
                        overall TEXT,
                        embedding VECTOR({get_settings().EMBEDDING_DIM}),
                        fts_vector tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(overall, ''))
                        ) STORED
                    );
                """)

                # Create Indexes
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{coll_name}_embedding_hnsw ON {coll_name} USING hnsw (embedding vector_cosine_ops);")
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{coll_name}_fts_gin ON {coll_name} USING gin (fts_vector);")
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{coll_name}_skills_gin ON {coll_name} USING gin (skills);")

                logger.info(f"Postgres collection {coll_name} initialized successfully.")
            except Exception as e:
                logger.error(f"Postgres initialization error: {e}", exc_info=True)
                raise

    async def save(self, data: DataFrame):
        """Persists data into Postgres using asyncpg."""
        settings = get_settings()
        batch_size = settings.BATCH_SIZE
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                batch_data = []
                for _, row in data.iterrows():
                    # Handle skills conversion: "Python, SQL" -> ["Python", "SQL"]
                    skills = row.get("Skills", [])
                    if isinstance(skills, str):
                        skills = [s.strip() for s in skills.split(",") if s.strip()]
                    elif not isinstance(skills, list):
                        skills = []

                    overall_text = row.get("overall", "")

                    # Convert embedding to list of floats for asyncpg
                    emb = row["embeddings"]
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()

                    batch_data.append((
                        str(row.get("ResumeID")), row.get("Name"), row.get("Category"),
                        row.get("Education"), skills, row.get("Summary"), row.get("Phone"), row.get("Location"),
                        overall_text, emb
                    ))

                    if len(batch_data) >= batch_size:
                        await self._execute_batch(conn, batch_data)
                        batch_data = []

                if batch_data:
                    await self._execute_batch(conn, batch_data)

                logger.info(f"Successfully saved {len(data)} records to Postgres.")
            except Exception as e:
                logger.error(f"Postgres save error: {e}", exc_info=True)
                raise

    async def _execute_batch(self, conn, batch):
        """Helper to execute a batch of inserts using executemany."""
        # Convert the embedding list to a string format '[v1, v2, ...]' for pgvector
        formatted_batch = [
            (*item[:-1], str(item[-1])) for item in batch
        ]
        query = f"""
            INSERT INTO {self.collection_name} 
            (resume_id, name, category, education, skills, summary, phone, location, overall, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector)
            ON CONFLICT (resume_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                overall = EXCLUDED.overall,
                summary = EXCLUDED.summary,
                skills = EXCLUDED.skills;
        """
        await conn.executemany(query, formatted_batch)

    async def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        """Standard semantic search using asyncpg."""
        results = []
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                # Convert embedding to string format for pgvector
                if hasattr(query_embedding, "tolist"):
                    emb_str = str(query_embedding.tolist())
                else:
                    emb_str = str(list(query_embedding))

                rows = await conn.fetch(
                    f"""
                    SELECT resume_id, name, category, education, skills, summary, phone, location,overall,
                           (embedding <=> $1::vector) as distance
                    FROM {self.collection_name}
                    ORDER BY distance ASC
                    LIMIT $2;
                    """,
                    emb_str, n_results
                )

                if not rows:
                    return {"results": []}

                # Prepare for reranking
                descriptions = [row["overall"] for row in rows]
                rerank_scores = self.reranker.rerank(query, descriptions)

                for row_data, r_score in zip(rows, rerank_scores):
                    # row_data is a Record object, convert to dict
                    row = dict(row_data)
                    distance = row.pop("distance")
                    row.pop("overall", None)  # Remove large overall text from response

                    dense_score = 1.0 - float(distance)
                    combined_score = dense_score + float(r_score)
                    results.append({
                        "payload": row,
                        "dense_score": dense_score,
                        "rerank_score": float(r_score),
                        "final_score": combined_score
                    })

                results.sort(key=lambda x: x["final_score"], reverse=True)
                return {"results": results}
            except Exception as e:
                logger.error(f"Postgres query error: {e}", exc_info=True)
                raise

    async def hybrid_search(self, query_embedding: Sequence[float],
                            n_results: int = 3, query: str = '') -> dict[str, list[Any]]:
        """Hybrid Search using Reciprocal Rank Fusion (RRF)."""
        if hasattr(query_embedding, "tolist"):
            emb_str = str(query_embedding.tolist())
        else:
            emb_str = str(list(query_embedding))

        sql = f"""
        WITH vector_search AS (
            SELECT resume_id, row_number() OVER (ORDER BY embedding <=> $1::vector) as rank
            FROM {self.collection_name}
            ORDER BY embedding <=> $1::vector
            LIMIT $2 * 4
        ),
        text_search AS (
            SELECT resume_id, row_number() OVER (ORDER BY ts_rank(fts_vector, websearch_to_tsquery('english', $3)) DESC) as rank
            FROM {self.collection_name}
            WHERE fts_vector @@ websearch_to_tsquery('english', $3)
            LIMIT $2 * 4
        )
        SELECT 
            r.id, r.resume_id, r.name, r.category, r.education, r.skills, r.summary,r.phone, r.location,
            (COALESCE(1.0 / (60 + vs.rank), 0.0) + COALESCE(1.0 / (60 + ts.rank), 0.0)) as rrf_score
        FROM vector_search vs
        FULL OUTER JOIN text_search ts ON vs.resume_id = ts.resume_id
        JOIN {self.collection_name} r ON r.resume_id = COALESCE(vs.resume_id, ts.resume_id)
        ORDER BY rrf_score DESC
        LIMIT $2;
        """
        results = []
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(sql, emb_str, n_results, query)
                for row_data in rows:
                    row = dict(row_data)
                    results.append({
                        "payload": row,
                        "final_score": float(row.pop("rrf_score"))
                    })
                return {"results": results}
            except Exception as e:
                logger.error(f"Postgres hybrid search error: {e}", exc_info=True)
                raise

    async def delete_collection(self) -> Optional[str]:
        """Drops the table."""
        coll_name = self.collection_name
        validate_collection_name(coll_name)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(f"DROP TABLE IF EXISTS {coll_name};")
                return coll_name
            except Exception as e:
                logger.error(f"Postgres delete error: {e}", exc_info=True)
                raise
        return None

    async def list_collection(self) -> list[str]:
        """Lists available tables in the public schema."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                return [r["table_name"] for r in rows]
            except Exception as e:
                logger.error(f"Postgres list error: {e}", exc_info=True)
                raise
