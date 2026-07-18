import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.whisper_service import whisper_service
from services.memory_service import memory_service
from services.llm_service import llm_service
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torch
router = APIRouter()

# Initialize local lightweight VAD model
vad_model = load_silero_vad()

@router.websocket("/stream-ingest")
async def stream_ingest(websocket: WebSocket):
    await websocket.accept()
    print("Client voice stream session active.")
    
    full_audio_buffer = bytearray()
    vad_buffer = bytearray()
    
    # 16kHz audio sampling parameters
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512  # Process ~32ms chunks
    BYTES_PER_SAMPLE = 2  # Int16
    CHUNK_BYTES = CHUNK_SIZE * BYTES_PER_SAMPLE
    
    silence_duration = 0.0
    SILENCE_THRESHOLD = 5.0  # Seconds
    is_processing = False

    try:
        while True:
            # Receive incoming continuous stream chunks from client
            data = await websocket.receive_bytes()

            full_audio_buffer.extend(data)
            vad_buffer.extend(data)
            
            # Keep processing chunks while they exist in the buffer
            while len(vad_buffer) >= CHUNK_BYTES:
                chunk = vad_buffer[:CHUNK_BYTES]
                del vad_buffer[:CHUNK_BYTES]
                
                # Convert raw bytes chunk to float32 normalized tensor array for Silero VAD
                audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

                audio_tensor = torch.from_numpy(audio_np)
                
                # Run lightweight local CPU VAD
                speech_probs = vad_model(audio_tensor, SAMPLE_RATE).item()
                
                if speech_probs < 0.4:  # User is silent
                    silence_duration += (CHUNK_SIZE / SAMPLE_RATE)
                else:  # User is speaking
                    silence_duration = 0.0
                
                # Trigger internal saving if user stops speaking for 5 seconds
                if silence_duration >= SILENCE_THRESHOLD and not is_processing:
                    is_processing = True
                    await websocket.send_json({"status": "warning", "message": "You aren't speaking. Saving memory now..."})
                    await asyncio.sleep(2.0)  # Visual alert hold time
                    raise WebSocketDisconnect  # Force loop break to processing pipeline

    except WebSocketDisconnect:
        print("Finalizing audio transcription pipeline...")
        print(f"DEBUG: full_audio_buffer length is {len(full_audio_buffer)}") # This will likely print 0
        
        if len(full_audio_buffer) > 0:
            print("Inside audio buffer>0")
            raw_audio = np.frombuffer(full_audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 1. Transcribe text
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, lambda: whisper_service.transcribe_raw_pcm(raw_audio)
            )

            print(f"text is {text}")
            
            if text and text.strip():
                print(f"Transcript generated: {text}")
                
                # 2. Extract context nodes
                try:
                    metadata = await llm_service.extract_memory_metadata(text)
                    print(f"LLM response is {metadata}")
                    entities = metadata.get("entities", [])
                    category = metadata.get("category", "general")
                except Exception as e:
                    print(f"LLM Metadata extraction failed: {e}")
                    raise e
                
                # 3. Commit to database 
                full_entry = await memory_service.save_text(text, entities=entities, category=category)
                print("SUCCESS: Memory indexed safely into Jarvis graph.")
                
            else:
                print("WARNING: No voice patterns detected in transcription.")
        else:
            print("WARNING: Audio buffer was empty upon disconnect. No data was received.")