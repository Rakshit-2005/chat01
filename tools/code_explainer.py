from tools.hf_infer import hf_text


def explain_code(code_text: str) -> str:
    prompt = f"""
You are an expert programmer. Given the code below, do the following in plain text:

1) Briefly explain what the code does (2-4 sentences).
2) Point out any obvious bugs, runtime errors, or security issues (if none, say 'No obvious bugs found').
3) Give a best-effort estimate of the time and space complexity of the core algorithm(s).
4) If code language is unclear, identify the most likely language.

Code:
```
{code_text[:8000]}
```
"""
    resp = hf_text("llama-3.1-8b-instant", prompt, max_length=600)
    if resp == "GROQ_TOKEN_MISSING":
        return "Code explanation skipped: GROQ_API_KEY not set."
    return resp
