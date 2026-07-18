import asyncio
from faster_whisper import WhisperModel
from config import settings

class WhisperService:
    def __init__(self):
        self.model = None

    def load_model(self):
        print(f"Loading Whisper Model ({settings.WHISPER_MODEL_SIZE}) onto {settings.WHISPER_DEVICE}...")
        self.model = WhisperModel(
            settings.WHISPER_MODEL_SIZE, 
            device=settings.WHISPER_DEVICE, 
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
        print("Whisper Model Loaded Successfully.")

    async def transcribe_audio(self, audio_array) -> str:
        """Runs transcription inside a thread pool to avoid blocking the async event loop."""
        if not self.model:
            raise RuntimeError("Whisper model is not loaded!")
            
        loop = asyncio.get_running_loop()
        # Offload CPU-bound inference execution
        segments, _ = await loop.run_in_executor(
            None, 
            lambda: self.model.transcribe(audio_array, beam_size=3)
        )
        
        transcript = " ".join([segment.text for segment in segments]).strip()
        return transcript

whisper_service = WhisperService()