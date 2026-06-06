import os
import base64
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _headers() -> dict:
    token = os.getenv("GROQ_API_KEY")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _has_token() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _hf_headers() -> dict:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _has_hf_token() -> bool:
    return bool(os.getenv("HUGGINGFACE_API_TOKEN"))


def hf_text(model: str, prompt: str, max_length: int = 512) -> str:
    if not _has_token():
        return "GROQ_TOKEN_MISSING"
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_length,
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=_headers(), json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices and isinstance(choices, list):
                first = choices[0] or {}
                message = first.get("message", {})
                if isinstance(message, dict) and message.get("content"):
                    return message["content"]
            if isinstance(data, dict) and data.get("error"):
                return f"GROQ_ERROR: {data.get('error')}"
            return str(data)
    except Exception as e:
        return f"GROQ_REQUEST_ERROR: {str(e)}"


def hf_image(model: str, image_bytes: bytes, content_type: Optional[str] = "image/png") -> str:
    if not _has_hf_token():
        return "HF_TOKEN_MISSING"
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        files = {"file": ("image", image_bytes, content_type)}
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, headers=_hf_headers(), files=files)
            r.raise_for_status()
            data = r.json()
            # often returns [{'generated_text': '...'}] or direct text
            if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and data.get("error"):
                return f"HF_ERROR: {data.get('error')}"
            return str(data)
    except Exception as e:
        return f"HF_IMAGE_ERROR: {str(e)}"


def hf_audio(model: str, audio_bytes: bytes, content_type: Optional[str] = "audio/wav") -> str:
    if not _has_hf_token():
        return "HF_TOKEN_MISSING"
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        headers = _hf_headers()
        headers["Content-Type"] = content_type
        with httpx.Client(timeout=180.0) as client:
            r = client.post(url, headers=headers, content=audio_bytes)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
                return data[0]["text"]
            if isinstance(data, dict) and data.get("error"):
                return f"HF_ERROR: {data.get('error')}"
            return str(data)
    except Exception as e:
        return f"HF_AUDIO_ERROR: {str(e)}"


def groq_vision(model: str, image_bytes: bytes, prompt: str, content_type: Optional[str] = "image/png") -> str:
    if not _has_token():
        return "GROQ_TOKEN_MISSING"
    url = "https://api.groq.com/openai/v1/chat/completions"
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, headers=_headers(), json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices and isinstance(choices, list):
                first = choices[0] or {}
                message = first.get("message", {})
                if isinstance(message, dict) and message.get("content"):
                    return message["content"]
            if isinstance(data, dict) and data.get("error"):
                return f"GROQ_ERROR: {data.get('error')}"
            return str(data)
    except Exception as e:
        return f"GROQ_VISION_ERROR: {str(e)}"


def hf_chat_completions(model: str, messages: list, max_length: int = 512) -> str:
    if not _has_token():
        return "GROQ_TOKEN_MISSING"
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_length,
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=_headers(), json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices and isinstance(choices, list):
                first = choices[0] or {}
                message = first.get("message", {})
                if isinstance(message, dict) and message.get("content"):
                    return message["content"]
            if isinstance(data, dict) and data.get("error"):
                return f"GROQ_ERROR: {data.get('error')}"
            return str(data)
    except Exception as e:
        return f"GROQ_REQUEST_ERROR: {str(e)}"
