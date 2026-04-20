from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import uvicorn
from app.services.agent import clinical_search_stream

from contextlib import asynccontextmanager
from app.services.scheduler import start_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    # Could gracefully shutdown scheduler here

app = FastAPI(
    title="Clinsight API",
    description="Next-Generation Medical Case Retrieval System Backend",
    version="1.0.0",
    lifespan=lifespan
)

from app.api.patients import router as patients_router
from app.api.twilio import router as twilio_router
from app.api.whatsapp import router as whatsapp_router
app.include_router(patients_router)
app.include_router(twilio_router)
app.include_router(whatsapp_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import List, Dict, Any

class QueryRequest(BaseModel):
    messages: List[Dict[str, Any]]
    filters: dict = {}

class FeedbackRequest(BaseModel):
    query: str
    case_id: str
    is_relevant: bool

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/search")
async def search_endpoint(request: QueryRequest):
    """
    Kicks off the robust LangGraph orchestrator stream
    """
    from app.services.agent_orchestrator import orchestrator_stream
    return StreamingResponse(
        orchestrator_stream(request.messages),
        media_type="text/event-stream"
    )

@app.post("/api/v1/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Captures the Doctor's Continuous Improvement training signal
    """
    feedback_file = "app/data/feedback_loop.json"
    feedback_data = []
    
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            try:
                feedback_data = json.load(f)
            except:
                pass
                
    feedback_data.append({
        "query": request.query,
        "case_id": request.case_id,
        "is_relevant": request.is_relevant
    })
    
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)
        
    return {"status": "success", "recorded": True}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
