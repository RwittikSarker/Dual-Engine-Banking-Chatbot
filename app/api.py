"""
api.py
------
FastAPI REST backend for the Dual-Engine Banking Chatbot.
Serves the prediction pipeline as an HTTP API consumed by the Next.js frontend.

Start:
    uvicorn app.api:app --reload --port 8000

Routes:
    POST /api/predict   — run the full dual-engine prediction pipeline
    GET  /api/health    — liveness probe
    GET  /api/examples  — list sidebar example queries
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import time
import asyncio
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from functools import lru_cache

from src.utils import get_logger, load_config

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Dual-Engine Banking Chatbot API",
    description="Classification-guided RAG pipeline for banking customer support.",
    version="1.0.0",
)

# Allow Next.js dev server (port 3000) and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pipeline startup — load models when server starts
# ──────────────────────────────────────────────

# Live status dict — updated as each model loads
_load_status: dict = {
    "ready": False,
    "stage": "idle",
    "steps": [],
    "error": None,
    "started_at": None,
    "ready_at": None,
}
_pipeline = None


def _warmup():
    """Load all models in a background thread at startup."""
    global _pipeline
    _load_status["started_at"] = time.strftime("%H:%M:%S")
    _load_status["stage"] = "starting"
    _load_status["steps"] = []

    def _step(msg: str):
        logger.info("[STARTUP] %s", msg)
        _load_status["steps"].append(msg)
        _load_status["stage"] = msg

    try:
        _step("⏳ Loading config...")
        cfg = load_config(str(_REPO_ROOT / "configs" / "config.yaml"))

        _step("⏳ Loading label encoder...")
        from src.utils import load_label_encoder
        load_label_encoder(cfg["paths"]["label_encoder"])

        _step("⏳ Loading DistilBERT intent classifier...")
        from src.predict import DualEnginePipeline

        _step("⏳ Loading MiniLM embedding model (GPU)...")
        # DualEnginePipeline loads everything internally
        _step("⏳ Loading FAISS vector index...")
        _step("⏳ Loading Flan-T5 generator (GPU)...")

        _pipeline = DualEnginePipeline(cfg)

        _load_status["ready"] = True
        _load_status["ready_at"] = time.strftime("%H:%M:%S")
        _step("✅ All models loaded — API is ready!")
    except Exception as exc:
        _load_status["error"] = str(exc)
        _load_status["stage"] = f"❌ Error: {exc}"
        logger.exception("Startup warmup failed: %s", exc)


@app.on_event("startup")
async def startup_event():
    """Trigger model loading in a background thread so the server stays responsive."""
    logger.info("=" * 60)
    logger.info("  Dual-Engine Banking Chatbot API starting...")
    logger.info("  Models will load in background — watch /api/status")
    logger.info("=" * 60)
    thread = threading.Thread(target=_warmup, daemon=True)
    thread.start()


def get_pipeline():
    """Return the cached pipeline (raises 503 if still loading)."""
    if _load_status["error"]:
        raise HTTPException(status_code=500, detail=f"Pipeline failed to load: {_load_status['error']}")
    if not _load_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Models still loading: {_load_status['stage']}. Check /api/status for progress."
        )
    return _pipeline


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User's banking query")


class PredictResponse(BaseModel):
    generated_answer: str
    rag_answer: str
    detected_intent: str
    confidence_score: float
    retrieval_similarity: float
    escalation_recommendation: str
    escalation_reasons: list[str]


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    stage: str
    steps: list[str]
    started_at: str | None
    ready_at: str | None
    error: str | None


class StatusResponse(BaseModel):
    ready: bool
    stage: str
    steps: list[str]
    started_at: str | None
    ready_at: str | None
    error: str | None

class ExamplesResponse(BaseModel):
    examples: list[str]


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "My card was charged twice",
    "How do I cancel a direct debit?",
    "What is my credit limit?",
    "I need to update my PIN",
    "I received a suspicious email",
    "My account is frozen",
    "How do I get a refund?",
    "I want to dispute a transaction",
    "My card hasn't arrived yet",
    "How do I set up Apple Pay?",
    "I forgot my passcode",
    "My card was stolen",
]


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe — reports pipeline load status."""
    return HealthResponse(
        status="ok" if _load_status["ready"] else "loading",
        pipeline_loaded=_load_status["ready"],
        stage=_load_status["stage"],
        steps=_load_status["steps"],
        started_at=_load_status["started_at"],
        ready_at=_load_status["ready_at"],
        error=_load_status["error"],
    )


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def status():
    """
    Live model-loading status endpoint.
    Poll this while waiting for the first message to see what's loading.
    Refreshes every second in the browser: http://localhost:8000/api/status
    """
    return StatusResponse(**{k: _load_status[k] for k in StatusResponse.model_fields})


@app.get("/api/examples", response_model=ExamplesResponse, tags=["Chat"])
async def examples():
    """Return the list of example banking queries shown in the UI sidebar."""
    return {"examples": EXAMPLE_QUERIES}


@app.post("/api/predict", response_model=PredictResponse, tags=["Chat"])
async def predict(req: PredictRequest):
    """
    Run the full dual-engine prediction pipeline.

    - Engine 1: DistilBERT intent classifier
    - Engine 2: MiniLM + FAISS + Flan-T5 RAG pipeline
    - Escalation: rule-based safety gate
    - Response: intent-matched professional template
    """
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model files not found: {exc}. Run train.py and retrieval.py --build first.",
        )

    try:
        result = pipeline.predict(req.query)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return PredictResponse(
        generated_answer=result["generated_answer"],
        rag_answer=result.get("rag_answer", ""),
        detected_intent=result["detected_intent"],
        confidence_score=result["confidence_score"],
        retrieval_similarity=result["retrieval_similarity"],
        escalation_recommendation=result["escalation_recommendation"],
        escalation_reasons=result["escalation_reasons"],
    )

