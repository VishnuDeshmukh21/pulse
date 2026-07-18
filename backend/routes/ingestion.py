import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import settings
from services.whisper_service import whisper_service
from services.memory_service import memory_service

router = APIRouter()

@router.websocket("/stream-ingest")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client voice stream session active.")
    
    audio_buffer = bytearray()
    silence_timer = 0.0
    
    try:
        while True:
            try:
                # Expecting raw, downsampled 16kHz 16-bit Mono PCM bytes from browser
                data = await asyncio.wait_for(
                    websocket.receive_bytes(), 
                    timeout=settings.CHUNK_TIMEOUT
                )
                audio_buffer.extend(data)
                silence_timer = 0.0  # Reset timer because audio tokens are actively arriving
                
            except asyncio.TimeoutError:
                # Triggered when no network chunks arrive within the 200ms timeout window
                if len(audio_buffer) > 0:
                    silence_timer += settings.CHUNK_TIMEOUT
                    
                    # Target silence threshold reached
                    if silence_timer >= settings.SILENCE_THRESHOLD_SECONDS:
                        await websocket.send_json({
                            "status": "processing", 
                            "message": "You stopped speaking. Finalizing memory..."
                        })
                        
                        # Process buffer: Convert raw PCM bytes back to Float32 array for Whisper
                        raw_data = np.frombuffer(audio_buffer, dtype=np.int16)
                        audio_np = raw_data.astype(np.float32) / 32768.0
                        
                        # Transcribe asynchronously
                        transcript = await whisper_service.transcribe_audio(audio_np)
                        
                        if transcript:
                            saved_entry = await memory_service.save_text(transcript)
                            print(f"Saved to DB: {saved_entry}")
                            await websocket.send_json({
                                "status": "success", 
                                "transcript": transcript,
                                "message": "Voice note securely saved to local brain."
                            })
                        else:
                            await websocket.send_json({
                                "status": "ignored", 
                                "message": "Silence detected. Buffer cleared."
                            })
                        
                        # Reset memory buffer indicators
                        audio_buffer.clear()
                        silence_timer = 0.0

    except WebSocketDisconnect:
        print("Client closed voice stream session.")