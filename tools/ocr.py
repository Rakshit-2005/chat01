import fitz  # PyMuPDF
from tools.hf_infer import groq_vision


def extract_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            txt = page.get_text()
            full_text += txt

        # If text is too short, it's likely a scanned PDF — use HF image OCR
        if len(full_text.strip()) < 100:
            full_text = ocr_scanned_pdf(doc)

        return full_text.strip()
    except Exception as e:
        return f"PDF extraction error: {str(e)}"


def ocr_scanned_pdf(doc) -> str:
    texts = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        resp = _ocr_image_bytes(img_bytes, "image/png")
        texts.append(resp)
    return "\n\n".join(texts)


def extract_from_image(file_bytes: bytes, filename: str) -> str:
    try:
        content_type = "image/png"
        if filename.lower().endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        return _ocr_image_bytes(file_bytes, content_type)
    except Exception as e:
        return f"Image OCR error: {str(e)}"


def _ocr_image_bytes(image_bytes: bytes, content_type: str) -> str:
    prompt = "Read the text from this image. Return only the extracted text, no extra commentary."
    resp = groq_vision("meta-llama/llama-4-scout-17b-16e-instruct", image_bytes, prompt, content_type=content_type)
    if resp == "GROQ_TOKEN_MISSING":
        return "Image OCR unavailable: GROQ_API_KEY is not set."
    if resp.startswith("GROQ_VISION_ERROR") or resp.startswith("GROQ_ERROR"):
        return "Image OCR unavailable right now on this deployment."
    return resp