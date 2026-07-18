import os

class Settings:
    # Model Configurations
    WHISPER_MODEL_SIZE: str = "base.en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    
    # Storage
    CHROMA_DB_PATH: str = "./jarvis_memory"
    CHROMA_COLLECTION_NAME: str = "jarvis_brain"
    
    # Audio Ingestion Settings
    SAMPLE_RATE: int = 16000  # Whisper expects 16kHz
    SILENCE_THRESHOLD_SECONDS: float = 4.0  # Time to wait before finalizing memory
    CHUNK_TIMEOUT: float = 0.2  # 200ms incoming WebSocket intervals

settings = Settings()