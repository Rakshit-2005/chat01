import os
from tools.hf_infer import groq_audio

def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """Transcribe audio using Groq's Whisper API."""
    ext = filename.split(".")[-1].lower()
    content_type = "audio/wav"
    if ext == "mp3":
        content_type = "audio/mpeg"
    elif ext == "m4a":
        content_type = "audio/mp4"
    elif ext == "wav":
        content_type = "audio/wav"

    transcript = groq_audio(file_bytes, filename, content_type)
    if transcript == "GROQ_TOKEN_MISSING":
        return "Audio transcription skipped: GROQ_API_KEY not set in environment."
    return transcript