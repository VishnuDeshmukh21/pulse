import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import settings
from services.whisper_service import whisper_service
from services.memory_service import memory_service

router = APIRouter()

@router.websocket("/stream-ingest")
async def stream_ingest(websocket: WebSocket):
    await websocket.accept()
    print("Client voice stream session active.")
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()

            audio_buffer.extend(data)
            
    except WebSocketDisconnect:
        print("Client closed voice stream session. Processing recording...")
        

        try:
           

            raw_audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            print(f" Converted to {len(raw_audio)} raw audio samples.")

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, 
                lambda: whisper_service.transcribe_raw_pcm(raw_audio)
            )

            if text and text.strip():
                print(f"Transcript: \"{text}\"")
                full_entry = await memory_service.save_text(text)
                print(f"Memory Pipeline Saved: {full_entry}")
            else:
                print("Whisper finished but found no clear speech in the recording.")

        except Exception as e:
            print(f"Error processing audio stream: {e}")
            