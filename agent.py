from tools.ocr import extract_from_pdf, extract_from_image
from tools.audio import transcribe_audio
from tools.youtube import fetch_youtube_transcript
from tools.summarizer import summarize_text
from tools.sentiment import analyze_sentiment
from tools.code_explainer import explain_code
from tools.hf_infer import hf_text, hf_chat_completions
import re
import json

SYSTEM_PROMPT = """
You are a smart AI agent that receives extracted content from files and a user query.

Your job:
1. Understand what the user wants
2. If the goal is unclear, reply ONLY with: CLARIFY: <your question>
3. If clear, reply with: TASK: <one of: summarize | sentiment | explain_code | answer | compare>

Only output the TASK or CLARIFY line. Nothing else.
"""

async def run_agent(query: str, files: list, history: str = "[]") -> dict:
    try:
        chat_history = json.loads(history)
    except Exception:
        chat_history = []
        
    steps = []
    extracted_texts = []

    # Step 1: Extract content from all files
    for file in files:
        fname = file["filename"].lower()
        steps.append(f"📄 Processing file: {file['filename']}")

        if fname.endswith(".pdf"):
            text = extract_from_pdf(file["bytes"])
            extracted_texts.append({"source": file["filename"], "text": text})
            steps.append(f"✅ PDF extracted: {len(text)} characters")

        elif fname.endswith((".jpg", ".jpeg", ".png")):
            text = extract_from_image(file["bytes"], file["filename"])
            extracted_texts.append({"source": file["filename"], "text": text})
            steps.append(f"✅ Image OCR done: {len(text)} characters")

        elif fname.endswith((".mp3", ".wav", ".m4a")):
            text = transcribe_audio(file["bytes"], file["filename"])
            extracted_texts.append({"source": file["filename"], "text": text})
            steps.append(f"✅ Audio transcribed: {len(text)} characters")

    # Step 2: Combine all extracted text + query
    combined_context = query + "\n\n"
    for item in extracted_texts:
        combined_context += f"[From {item['source']}]:\n{item['text']}\n\n"

    # Step 3: Check for YouTube URLs
    youtube_pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]+)'
    yt_urls = re.findall(youtube_pattern, combined_context)
    yt_texts = []
    for url in yt_urls:
        steps.append(f"🎥 YouTube URL detected: {url}")
        yt_text = fetch_youtube_transcript(url)
        yt_texts.append({"source": url, "text": yt_text})
        combined_context += f"[YouTube transcript from {url}]:\n{yt_text}\n\n"
        steps.append(f"✅ YouTube transcript fetched")

    # Step 4: Identify intent locally
    steps.append("🧠 Agent identifying intent (rule-based)...")
    intent_raw = classify_intent_local(query, extracted_texts, combined_context)

    # Step 5: Handle clarification
    if intent_raw.startswith("CLARIFY:"):
        question = intent_raw.replace("CLARIFY:", "").strip()
        steps.append(f"❓ Clarification needed")
        return {
            "status": "clarify",
            "question": question,
            "steps": steps,
            "extracted_texts": extracted_texts
        }

    # Step 6: Extract task
    task = "answer"
    if intent_raw.startswith("TASK:"):
        task = intent_raw.replace("TASK:", "").strip().lower()
    steps.append(f"🎯 Task identified: {task}")

    # Step 7: Execute the task
    output = await execute_task(task, combined_context, query, steps, chat_history)

    return {
        "status": "success",
        "task": task,
        "output": output,
        "steps": steps,
        "extracted_texts": [{"source": i["source"], "text": i["text"][:500]} for i in extracted_texts]
    }


async def execute_task(task: str, context: str, query: str, steps: list, chat_history: list) -> str:
    steps.append(f"⚙️ Executing: {task}")

    if task == "summarize":
        return summarize_text(context)

    elif task == "sentiment":
        return analyze_sentiment(context)

    elif task == "explain_code":
        return explain_code(context)

    elif task == "compare":
        return compare_inputs(context)

    else:  # general answer
        return answer_question(query, context, chat_history)


def compare_inputs(context: str) -> str:
    prompt = f"""
Compare the items in the context below. Reply with a concise comparison in plain text.

Context:
{context[:6000]}
"""
    resp = hf_text("llama-3.1-8b-instant", prompt, max_length=300)
    if resp == "GROQ_TOKEN_MISSING":
        return "Comparison unavailable: GROQ_API_KEY not set."
    if resp.startswith("GROQ_"):
        return "Comparison unavailable right now."
    return resp


def classify_intent_local(query: str, extracted_texts: list, combined_context: str) -> str:
    q = query.lower().strip()
    if not q:
        if extracted_texts:
            # Auto-default to summarizing uploaded files if no specific query is provided
            return "TASK: summarize"
        return "CLARIFY: What do you want me to do with the uploaded content?"

    if any(word in q for word in ["summarize", "summary"]):
        return "TASK: summarize"
    if "sentiment" in q:
        return "TASK: sentiment"
    if any(word in q for word in ["explain code", "explain this code", "what does this code do", "bug"]):
        return "TASK: explain_code"
    if "compare" in q or "same topic" in q:
        return "TASK: compare"
    if q.startswith("what is ") or q.startswith("define ") or q.startswith("explain ") or q.startswith("tell me about "):
        return "TASK: answer"

    if extracted_texts:
        return "TASK: answer"

    return "TASK: answer"

def answer_question(query: str, context: str, chat_history: list) -> str:
    system_content = f"""Answer the user's question using the provided context when relevant.

If the context is insufficient, answer based on general knowledge.
Reply in plain text and keep it concise.

Context:
{context[:6000]}"""

    messages = [{"role": "system", "content": system_content}]
    
    # Prepend history turns (limit to last 10 messages to keep context window clean)
    for msg in chat_history[-10:]:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
        
    # Append current query
    messages.append({"role": "user", "content": query})
    
    # Debug logging
    print("DEBUG - messages sent to Groq:", json.dumps(messages, indent=2))
    
    resp = hf_chat_completions("llama-3.1-8b-instant", messages, max_length=256)
    if resp == "GROQ_TOKEN_MISSING":
        return "Answer unavailable: GROQ_API_KEY not set."
    if resp.startswith("GROQ_"):
        return "Answer unavailable right now."
    return resp