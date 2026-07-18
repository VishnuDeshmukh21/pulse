import subprocess
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from services.whisper_service import whisper_service
from services.memory_service import memory_service
from routes import ingestion
from dotenv import load_dotenv

load_dotenv()
# --- LLM Configuration ---
LLAMA_DIR = os.getenv("LLAMA_DIR", "llama-server.exe")
SERVER_EXE = os.path.join(LLAMA_DIR, "llama-server.exe")
MODEL_PATH = os.path.join(LLAMA_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf")

llm_process = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup Event Logic ----
    global llm_process
    
    print("Initializing Memory DB...")
    await memory_service.init_db()
    
    print("Loading Whisper Model...")
    whisper_service.load_model()
    
    print("Starting local LLM server (llama.cpp)...")
    command = [
        SERVER_EXE,
        "-m", MODEL_PATH,
        "-c", "4096",        # Context window size
        "--port", "8080"     # Run LLM on port 8080 so it doesn't clash with FastAPI on 8000
    ]
    
    try:
        # Launch llama-server in the background
        llm_process = subprocess.Popen(command)
        print(f"LLM Server successfully started on port 8080 (PID: {llm_process.pid})")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{SERVER_EXE}'. Check if the executable is named 'server.exe' instead.")
        
    yield
    
    # ---- Shutdown Event Logic ----
    print("Cleaning and shutting down local Jarvis server.")
    
    # Shut down database connection
    await memory_service.close()
    
    # Shut down the LLM subprocess cleanlyc
    if llm_process:
        print("Shutting down llama.cpp server...")
        llm_process.terminate()
        llm_process.wait()

app = FastAPI(title="Jarvis Core Local Backend", lifespan=lifespan)
app.include_router(ingestion.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)