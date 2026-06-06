from tools.hf_infer import hf_text


def summarize_text(text: str) -> str:
    prompt = f"""
Summarize the following content in exactly this format:

ONE-LINE SUMMARY:
<single sentence summary>

KEY POINTS:
- <point 1>
- <point 2>
- <point 3>

DETAILED SUMMARY:
<exactly 5 sentences covering the main ideas>

Content:
{text[:6000]}
"""
    resp = hf_text("llama-3.1-8b-instant", prompt, max_length=300)
    if resp == "GROQ_TOKEN_MISSING":
        return "Summarization skipped: GROQ_API_KEY not set."
    return resp
