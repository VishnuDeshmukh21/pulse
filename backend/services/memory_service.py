import chromadb
import datetime
import time
from config import settings

class MemoryService:
    def __init__(self):
        self.client = None
        self.collection = None

    def init_db(self):
        print(f"🗄️ Initializing ChromaDB persistence at {settings.CHROMA_DB_PATH}...")
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME)
        print("✅ ChromaDB Connected.")

    async def save_text(self, text: str) -> str:
        """Saves text into ChromaDB with a localized structural timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_entry = f"[{timestamp}] {text}"
        memory_id = f"mem_{int(time.time())}"
        
        # ChromaDB operations are synchronous, offload if performance drops
        self.collection.add(
            documents=[full_entry],
            ids=[memory_id]
        )
        return full_entry

memory_service = MemoryService()