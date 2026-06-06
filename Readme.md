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

```mermaid
graph TD
    %% User Inputs
    subgraph User Interface (Browser)
        UI[index.html - Premium Chat UI]
        Files[Uploaded Files: PDF, Image, Audio]
        Text[User text queries]
        Memory[chatHistory state in browser]
    end

    %% Backend Entrypoint
    subgraph FastAPI Backend
        API[main.py - /agent endpoint]
        Core[agent.py - Orchestrator]
    end

    %% Tool Registry and Pipelines
    subgraph Tool Registry & Pipelines
        PDF[ocr.py - PyMuPDF Parser]
        OCR[ocr.py - Groq Llama 4 Vision OCR]
        Audio[audio.py - Groq Whisper STT]
        YT[youtube.py - YouTube Transcript API]
        LLM[hf_infer.py - Groq Llama 3 API]
    end

    %% Flow connections
    UI -->|Sends Form Data: query, files, history| API
    API -->|Calls run_agent| Core
    
    %% Processing Inputs
    Core -->|1. Processes PDF| PDF
    Core -->|2. Processes Image| OCR
    Core -->|3. Processes Audio| Audio
    Core -->|4. Detects YT URLs| YT
    
    %% Intent Classification
    Core -->|5. Local Intent Classifier| Intent{Rule-Based Classifier}
    
    %% Chaining and LLM execution
    Intent -->|A. Conversational Q&A| Qn[answer_question]
    Intent -->|B. Summarize / Sentiment / Code Explainer| Tasks[Task Executors]
    
    Qn -->|Includes Chat History| LLM
    Tasks -->|Context Analysis| LLM
    
    %% Returns response
    LLM -->|Generates Output| Core
    Core -->|Returns status, output, steps, full text| API
    API -->|Sends JSON Response| UI
    UI -->|Updates chatHistory and renders view| Memory
```

## Supported Tasks
- Summarization (1-line + bullets + 5-sentence)
- Sentiment Analysis
- Code Explanation
- Audio Transcription
- YouTube Transcript Fetching
- Cross-input Reasoning
- General Q&A

## Remaining Assignment Items
- [x] Architecture diagram (Added to Readme.md)
- [ ] Test cases (automated suite)
- [x] Audio transcribe + summary chaining (Implemented in agent.py)
- [x] OCR confidence reporting (Implemented in ocr.py)
- [x] Deployment URL for the final submission (Deployed on Render)