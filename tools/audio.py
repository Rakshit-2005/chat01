import os
from tools.hf_infer import hf_audio

def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """Transcribe audio using Hugging Face Whisper-style inference."""
    ext = filename.split(".")[-1].lower()
    content_type = "audio/wav"
    if ext == "mp3":
        content_type = "audio/mpeg"
    elif ext == "m4a":
        content_type = "audio/mp4"
    elif ext == "wav":
        content_type = "audio/wav"

    transcript = hf_audio("openai/whisper-small", file_bytes, content_type=content_type)
    if transcript == "HF_TOKEN_MISSING":
        return "Audio transcription skipped: HUGGINGFACE_API_TOKEN not set in environment."
    return transcript