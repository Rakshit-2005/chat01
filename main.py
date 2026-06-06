from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from typing import List, Optional
import uvicorn
from agent import run_agent
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multi-Modal AI Agent")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("frontend/index.html")

@app.post("/agent")
async def agent_endpoint(
    query: Optional[str] = Form(default=""),
    files: Optional[List[UploadFile]] = File(default=None)
):
    file_data = []
    if files:
        for file in files:
            content = await file.read()
            file_data.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "bytes": content
            })

    try:
        result = await run_agent(query=query, files=file_data)
        return result
    except Exception as e:
        # Ensure we always return JSON to the frontend (avoid HTML 500 pages)
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)