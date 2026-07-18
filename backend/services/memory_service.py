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
        
        print(f"🗄️ Connecting to PostgreSQL at {settings.PG_HOST}...")
        
        self.pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"row_factory": dict_row}
        )
        # Explicitly open the pool asynchronously
        await self.pool.open()
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # This requires pgvector to be available on your Postgres server
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                await register_vector_async(conn)
                
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS jarvis_memories (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        raw_text TEXT NOT NULL,
                        embedding vector(384),
                        graph_nodes JSONB DEFAULT '{}'::jsonb
                    );
                """)
            await conn.commit()
        print("PostgreSQL Vector Database Ready.")

    async def save_text(self, text: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        loop = asyncio.get_running_loop()
        embedding_array = await loop.run_in_executor(
            None, 
            lambda: self.embedding_model.encode(text).tolist()
        )
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO jarvis_memories (created_at, raw_text, embedding)
                    VALUES (%s, %s, %s)
                    RETURNING id, raw_text;
                """, (datetime.datetime.now(), text, embedding_array))
                
                result = await cur.fetchone()
            await conn.commit()
            
        full_entry = f"[{timestamp}] {result['raw_text']}"
        return full_entry

    async def close(self):
        if self.pool:
            await self.pool.close()

memory_service = MemoryService()