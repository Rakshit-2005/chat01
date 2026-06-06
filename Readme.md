# Multi-Modal AI Agent

An agentic application that accepts Text, Images, PDFs, and Audio files,
understands the user's goal, and autonomously performs the correct task.

## Setup

### 1. Clone the repo
git clone <your-repo-url>
cd <your-repo>

### 2. Create .env file
GROQ_API_KEY=your_groq_api_key_here
# Optional: only needed for audio transcription
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run locally
uvicorn main:app --reload

Visit: http://localhost:8000

## Deployment (Render.com)
See deployment steps below.

### Render note
Text answering and image OCR use Groq, while audio transcription still uses Hugging Face.
Keep both API keys in Render environment variables if you want all input modes enabled.

### Render tip
Do not deploy `.venv`, `.env`, or cache folders. The included `.dockerignore` keeps the build context small.

## Architecture
Architecture diagram still needs to be added in `/docs/architecture.png`.

## Supported Tasks
- Summarization (1-line + bullets + 5-sentence)
- Sentiment Analysis
- Code Explanation
- Audio Transcription
- YouTube Transcript Fetching
- Cross-input Reasoning
- General Q&A

## Remaining Assignment Items
- Architecture diagram
- Test cases
- Audio transcribe + summary chaining
- OCR confidence reporting
- Deployment URL for the final submission