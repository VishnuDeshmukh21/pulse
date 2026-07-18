from fastapi import FastAPI
from contextlib import asynccontextmanager
from services.whisper_service import whisper_service
from services.memory_service import memory_service
from routes import ingestion

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup Event Logic ----
    memory_service.init_db()
    whisper_service.load_model()
    yield
    # ---- Shutdown Event Logic ----
    print("🛑 Cleaning and shutting down local Jarvis server.")

app = FastAPI(title="Jarvis Core Local Backend", lifespan=lifespan)

app.include_router(ingestion.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    # Run server locally on standard port
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)