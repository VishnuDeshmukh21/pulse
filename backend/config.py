import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Model Configurations
    WHISPER_MODEL_SIZE: str = "small.en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    
    # Storage
    CHROMA_DB_PATH: str = "./jarvis_memory"
    CHROMA_COLLECTION_NAME: str = "jarvis_brain"
    
    # Audio Ingestion Settings
    SAMPLE_RATE: int = 16000  # Whisper expects 16kHz
    SILENCE_THRESHOLD_SECONDS: float = 4.0  # Time to wait before finalizing memory
    CHUNK_TIMEOUT: float = 0.2  # 200ms incoming WebSocket intervals

    PG_USER: str = os.getenv("PG_USER")
    PG_PASSWORD: str = os.getenv("PG_PASSWORD")
    PG_HOST: str =  os.getenv("PG_HOST")
    PG_PORT: str =  os.getenv("PG_PORT")
    PG_DB_NAME: str =  os.getenv("PG_DB_NAME")

    LLM_SERVER_URL: str = "http://127.0.0.1:8080/v1"
    LLM_MODEL_NAME: str = "qwen2.5-3b"

    @property
    def database_url(self):
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB_NAME}"

settings = Settings()