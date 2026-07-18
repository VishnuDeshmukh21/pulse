import datetime
import asyncio
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async
from sentence_transformers import SentenceTransformer
from config import settings

class MemoryService:
    def __init__(self):
        self.pool = None
        self.embedding_model = None

    async def init_db(self):
        print("Loading local embedding model (CPU)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        print(f"Connecting to PostgreSQL at {settings.PG_HOST}...")
        self.pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=1, max_size=5, open=False,
            kwargs={"row_factory": dict_row}
        )
        await self.pool.open()
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await register_vector_async(conn)
                
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS jarvis_memories (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        raw_text TEXT NOT NULL,
                        embedding vector(384),
                        entities TEXT[] DEFAULT '{}',
                        category VARCHAR(50) DEFAULT 'general'
                    );
                """)
                
                # Create an index for lightning-fast entity matching
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_entities 
                    ON jarvis_memories USING gin(entities);
                """)
            await conn.commit()
        print("PostgreSQL Vector/Graph Database Ready.")

    async def save_text(self, text: str, entities: list = None, category: str = "general") -> str:
        if entities is None:
            entities = []
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        loop = asyncio.get_running_loop()
        embedding_array = await loop.run_in_executor(
            None, 
            lambda: self.embedding_model.encode(text).tolist()
        )
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Inserting entities and category into the database
                await cur.execute("""
                    INSERT INTO jarvis_memories (created_at, raw_text, embedding, entities, category)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, raw_text;
                """, (datetime.datetime.now(), text, embedding_array, entities, category))
                
                result = await cur.fetchone()
            await conn.commit()
            
        full_entry = f"[{timestamp}] {result['raw_text']}"
        return full_entry

    async def search_memories(self, query: str, extracted_entities: list, days_limit: int = 7) -> list:
        """
        Fetches relevant memories based on hybrid vector + graph entity matching.
        """
        # Embed the user's vaguer search query
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None, 
            lambda: self.embedding_model.encode(query).tolist()
        )
        
        # Calculate the actual datetime cutoff instead of using SQL intervals directly
        # This is safer for psycopg parameter passing
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_limit)
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT id, raw_text, created_at, entities
                    FROM jarvis_memories
                    WHERE created_at >= %s
                      AND (
                          entities && %s::TEXT[]       -- Graph Entity Match
                          OR (embedding <=> %s) < 0.4  -- Semantic Vector Match
                      )
                    ORDER BY (embedding <=> %s) ASC
                    LIMIT 5;
                """, (cutoff_date, extracted_entities, query_embedding, query_embedding))
                
                results = await cur.fetchall()
                return results

    async def close(self):
        if self.pool:
            await self.pool.close()

memory_service = MemoryService()