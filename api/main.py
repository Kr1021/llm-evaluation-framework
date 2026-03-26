"""
api/main.py

FastAPI app exposing a /evaluate POST endpoint.
Accepts question, context, and answer, runs all three evaluator modules,
and returns a unified JSON response with every score.

Run locally:
    uvicorn api.main:app --reload

POST /evaluate
    {
        "question":    "What is the claims deadline?",
        "context":     "Claims must be submitted within 90 days.",
        "answer":      "You have 90 days to submit a claim.",
        "ground_truth": "Claims must be submitted within 90 days."  # optional
    }
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Evaluator imports
# ---------------------------------------------------------------------------
from evaluators.hallucination import score_hallucination, score_hallucination_sentences
from evaluators.output_scorer import OutputScorer

# RAGAS scorer is optional — skip gracefully if ragas is not installed
try:
    from evaluators.ragas_scorer import score_all as ragas_score_all
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Evaluation Framework API",
    description="Evaluate LLM outputs for hallucination, output quality, and RAGAS metrics.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    question: str = Field(..., description="The user question posed to the RAG system.")
    context: str  = Field(..., description="The retrieved context provided to the LLM.")
    answer: str   = Field(..., description="The generated answer to be evaluated.")
    ground_truth: Optional[str] = Field(
        None,
        description="Optional reference answer. Used by context_recall if provided.",
    )


class HallucinationScores(BaseModel):
    hallucination_score: float
    similarity: float
    verdict: str                  # "GROUNDED" | "HALLUCINATED"
    is_hallucination: bool
    threshold: float
    mode: str                     # "sentence_level"


class OutputScores(BaseModel):
    coherence: Optional[float]
    relevance: Optional[float]
    completeness: Optional[float]
    rouge_l: Optional[float]
    overall: float
    passed: bool


class RAGASScores(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_recall: float
    mode: str                     # "live" | "mock"
    available: bool


class EvaluateResponse(BaseModel):
    question: str
    context: str
    answer: str
    ground_truth: Optional[str]
    hallucination: HallucinationScores
    output_quality: OutputScores
    ragas: RAGASScores
    duration_seconds: float
    evaluator_mode: str           # "live" (OpenAI key set) | "mock" (no key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _run_hallucination(question: str, context: str, answer: str) -> HallucinationScores:
    """Run sentence-level hallucination scoring."""
    result = score_hallucination_sentences(answer=answer, context=context)
    return HallucinationScores(
        hallucination_score=result.hallucination_score,
        similarity=result.similarity,
        verdict=result.verdict,
        is_hallucination=result.is_hallucination,
        threshold=result.threshold,
        mode="sentence_level",
    )


def _run_output_scorer(question: str, context: str, answer: str, ground_truth: Optional[str]) -> OutputScores:
    """Run OutputScorer (TF-IDF relevance + optional ROUGE)."""
    scorer = OutputScorer()
    result = scorer.score(query=question, response=answer, reference=ground_truth)
    return OutputScores(
        coherence=result.scores.get("coherence"),
        relevance=result.scores.get("relevance"),
        completeness=result.scores.get("completeness"),
        rouge_l=result.scores.get("rouge_l"),
        overall=result.overall,
        passed=result.passed,
    )


def _run_ragas(question: str, context: str, answer: str, ground_truth: Optional[str]) -> RAGASScores:
    """Run RAGAS metrics if the library is installed."""
    if not _RAGAS_AVAILABLE:
        return RAGASScores(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_recall=0.0,
            mode="unavailable",
            available=False,
        )
    result = ragas_score_all(
        question=question,
        context=context,
        answer=answer,
        ground_truth=ground_truth,
    )
    return RAGASScores(
        faithfulness=result.faithfulness,
        answer_relevance=result.answer_relevance,
        context_recall=result.context_recall,
        mode=result.mode,
        available=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {
        "status": "ok",
        "service": "LLM Evaluation Framework API",
        "evaluator_mode": "live" if _has_api_key() else "mock",
        "ragas_available": _RAGAS_AVAILABLE,
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check."""
    return {
        "status": "ok",
        "openai_key_set": _has_api_key(),
        "ragas_available": _RAGAS_AVAILABLE,
    }


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate(request: EvaluateRequest):
    """
    Run all evaluators on a (question, context, answer) triple and return
    every score as a unified JSON response.

    - **hallucination**: sentence-level cosine similarity score (no API needed)
    - **output_quality**: coherence, relevance, completeness, ROUGE-L
    - **ragas**: faithfulness, answer_relevance, context_recall
      (uses mock LLM when OPENAI_API_KEY is not set; skipped entirely
       if the ragas package is not installed)
    """
    start = time.perf_counter()

    try:
        hallucination_scores = _run_hallucination(
            request.question, request.context, request.answer
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hallucination scorer failed: {exc}")

    try:
        output_scores = _run_output_scorer(
            request.question, request.context, request.answer, request.ground_truth
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Output scorer failed: {exc}")

    try:
        ragas_scores = _run_ragas(
            request.question, request.context, request.answer, request.ground_truth
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAGAS scorer failed: {exc}")

    duration = round(time.perf_counter() - start, 4)

    return EvaluateResponse(
        question=request.question,
        context=request.context,
        answer=request.answer,
        ground_truth=request.ground_truth,
        hallucination=hallucination_scores,
        output_quality=output_scores,
        ragas=ragas_scores,
        duration_seconds=duration,
        evaluator_mode="live" if _has_api_key() else "mock",
    )


@app.post("/evaluate/hallucination", tags=["Evaluation"])
def evaluate_hallucination(request: EvaluateRequest):
    """Run only the hallucination scorer."""
    try:
        result = score_hallucination_sentences(
            answer=request.answer, context=request.context
        )
        return {
            "hallucination_score": result.hallucination_score,
            "similarity": result.similarity,
            "verdict": result.verdict,
            "is_hallucination": result.is_hallucination,
            "threshold": result.threshold,
            "sentence_scores": result.sentence_scores,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate/output", tags=["Evaluation"])
def evaluate_output(request: EvaluateRequest):
    """Run only the output quality scorer."""
    try:
        scorer = OutputScorer()
        result = scorer.score(
            query=request.question,
            response=request.answer,
            reference=request.ground_truth,
        )
        return {
            "scores": result.scores,
            "overall": result.overall,
            "passed": result.passed,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

---

## Step 2 — Edit `requirements.txt`

Open your existing `requirements.txt` → click the **pencil (edit) icon** → add these two lines at the bottom:
```
fastapi==0.111.0
uvicorn==0.29.0
```

Your final `requirements.txt` will look like:
```
langchain==0.1.16
langchain-openai==0.1.3
openai==1.23.2
sentence-transformers==2.7.0
numpy==1.26.4
pandas==2.2.2
rouge-score==0.1.2
boto3==1.34.84
python-dotenv==1.0.1
pytest==8.1.1
pytest-asyncio==0.23.6
pydantic==2.7.1
fastapi==0.111.0
uvicorn==0.29.0
```
