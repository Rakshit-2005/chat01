from tools.hf_infer import hf_text


def analyze_sentiment(text: str) -> str:
    prompt = f"""
Analyze the sentiment of the following text. Reply in exactly this format:

SENTIMENT: <Positive / Negative / Neutral / Mixed>
CONFIDENCE: <percentage, e.g. 87%>
JUSTIFICATION: <one sentence explaining why>

Text:
{text[:4000]}
"""
    resp = hf_text("llama-3.1-8b-instant", prompt, max_length=128)
    if resp == "GROQ_TOKEN_MISSING":
        return "Sentiment skipped: GROQ_API_KEY not set."
    return resp