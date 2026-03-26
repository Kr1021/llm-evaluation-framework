# LLM Evaluation Framework

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-0.1.9-blueviolet?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1.16-brightgreen?logo=chainlink&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Lambda-orange?logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-black?logo=githubactions&logoColor=white)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> **Production-grade end-to-end LLM evaluation framework** for measuring output
> reliability, detecting hallucinations, and automating testing workflows for
> Generative AI systems deployed in regulated environments.

---

## Table of Contents

- [The Problem](#the-problem)
- [What This Framework Does](#what-this-framework-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Running Locally](#running-locally)
- [API Reference](#api-reference)
- [How CI Works](#how-ci-works)
- [Sample Report](#sample-report)
- [Governance & Compliance](#governance--compliance)
- [Author](#author)

---

## The Problem

Organisations deploying LLMs in regulated industries — **insurance, finance,
healthcare** — face a critical gap:

> *Model outputs are probabilistic. Compliance obligations are deterministic.*

Without systematic evaluation, teams have no auditable evidence that their LLM:
- Is grounded in the source context it was given (not hallucinating)
- Produces outputs that are consistent across prompt versions (no silent regressions)
- Meets the reliability bar required for production deployment

Manual review does not scale. A compliance team reviewing 500 test cases per sprint
is a bottleneck, not a safety net.

---

## What This Framework Does

| Capability | Module | API needed? |
|---|---|---|
| Hallucination detection via sentence similarity | `evaluators/hallucination.py` | No |
| Hallucination detection via LLM claim verification | `evaluators/hallucination_detector.py` | Optional |
| Output quality scoring (relevance, coherence, ROUGE-L) | `evaluators/output_scorer.py` | Optional |
| RAGAS metrics (faithfulness, answer relevance, context recall) | `evaluators/ragas_scorer.py` | Optional |
| Prompt regression testing against a YAML baseline | `evaluators/prompt_regression.py` | Optional |
| REST API exposing all scores as JSON | `api/main.py` | — |
| Full evaluation pipeline with S3 export | `pipelines/eval_pipeline.py` | Optional |
| CI evaluation pipeline with JSON report | `pipelines/ci_eval_pipeline.py` | No |

**Every module has a mock/offline fallback.** The entire framework runs without
an `OPENAI_API_KEY` — CI never breaks, local dev requires zero credentials.

---

## Architecture
```
 Input (JSON / CSV / API request)
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EvaluationPipeline                         │
│                                                                 │
│  ┌──────────────────────┐   ┌──────────────────────────────┐   │
│  │  HallucinationScorer │   │       OutputScorer           │   │
│  │                      │   │                              │   │
│  │  • Sentence cosine   │   │  • Answer relevance (TF-IDF) │   │
│  │    similarity        │   │  • Coherence (LLM / mock)    │   │
│  │  • Claim-level NLI   │   │  • Completeness (LLM / mock) │   │
│  │  • Verdict string    │   │  • ROUGE-L (vs ground truth) │   │
│  └──────────────────────┘   └──────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────┐   ┌──────────────────────────────┐   │
│  │     RAGASScorer      │   │   PromptRegressionTester     │   │
│  │                      │   │                              │   │
│  │  • Faithfulness      │   │  • YAML test suite loader    │   │
│  │  • Answer relevancy  │   │  • TF-IDF cosine diff        │   │
│  │  • Context recall    │   │  • Regression flag per case  │   │
│  │  • Mock LLM fallback │   │  • Pass rate report          │   │
│  └──────────────────────┘   └──────────────────────────────┘   │
│                                                                 │
│                  Aggregated Result Dict                         │
└─────────────────────────────────────────────────────────────────┘
          │                          │
          ▼                          ▼
  JSON Report (local)         S3 Upload (optional)
  outputs/report.json         s3://bucket/eval-results/


  ┌──────────────────────────────────────────────────────────────┐
  │                      FastAPI  /evaluate                      │
  │  POST { question, context, answer, ground_truth? }           │
  │  ← { hallucination{}, output_quality{}, ragas{}, … }        │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │                    GitHub Actions CI                         │
  │  push/PR → install deps → pytest → ci_eval_pipeline.py      │
  │                        → upload JSON artifact               │
  └──────────────────────────────────────────────────────────────┘
```

---

## Project Structure
```
llm-evaluation-framework/
│
├── evaluators/
│   ├── hallucination.py          # Sentence-similarity hallucination scorer
│   ├── hallucination_detector.py # LLM claim-verification detector
│   ├── output_scorer.py          # Relevance, coherence, ROUGE-L
│   ├── ragas_scorer.py           # RAGAS: faithfulness, relevance, recall
│   └── prompt_regression.py      # YAML-driven prompt regression tester
│
├── pipelines/
│   ├── eval_pipeline.py          # Full pipeline with S3 export
│   └── ci_eval_pipeline.py       # Lightweight CI pipeline → JSON report
│
├── api/
│   └── main.py                   # FastAPI /evaluate endpoint
│
├── tests/
│   ├── test_evaluators.py        # Pytest suite (fully offline)
│   ├── sample_eval_data.json     # 5 sample records for CI
│   └── prompt_suite.yaml         # YAML regression test suite
│
├── outputs/                      # Auto-created; holds generated reports
│
├── .github/
│   └── workflows/
│       └── eval-ci.yml           # GitHub Actions CI workflow
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **LLM Evaluation** | [RAGAS](https://github.com/explodinggradients/ragas) |
| **LLM Orchestration** | [LangChain](https://github.com/langchain-ai/langchain) |
| **Sentence Embeddings** | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| **NLP / Similarity** | scikit-learn TF-IDF cosine, ROUGE-L |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| **Cloud Storage** | AWS S3 (via boto3) |
| **Serverless** | AWS Lambda (pipeline deployment target) |
| **Testing** | Pytest + pytest-asyncio |
| **CI/CD** | GitHub Actions |
| **Containerisation** | Docker |
| **Config** | python-dotenv, PyYAML, Pydantic v2 |

---

## Quick Start
```python
from evaluators.hallucination import score_hallucination
from evaluators.ragas_scorer import score_all

result = score_hallucination(
    answer="You must submit claims within 90 days.",
    context="Claims must be submitted within 90 days of the service date.",
)
print(result.verdict)            # "GROUNDED"
print(result.hallucination_score)  # 0.04

ragas = score_all(
    question="What is the claims deadline?",
    context="Claims must be submitted within 90 days of the service date.",
    answer="You must submit claims within 90 days.",
)
print(ragas.faithfulness)       # 1.0 (mock) or real score with API key
print(ragas.mode)               # "mock" | "live"
```

---

## Running Locally

### 1. Clone & install
```bash
git clone https://github.com/Kr1021/llm-evaluation-framework.git
cd llm-evaluation-framework
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables (optional)
```bash
# Without this key all evaluators run in mock/offline mode — no errors
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 3. Run the evaluators directly
```bash
# Hallucination score (no key needed)
python -c "
from evaluators.hallucination import score_hallucination
r = score_hallucination(
    answer='Claims must be filed within 30 days.',
    context='Claims must be submitted within 90 days of the service date.'
)
print(r.verdict, r.hallucination_score)
"
```

### 4. Run the FastAPI server
```bash
uvicorn api.main:app --reload
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

Example request:
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the claims deadline?",
    "context":  "Claims must be submitted within 90 days of the service date.",
    "answer":   "You have 90 days to file a claim.",
    "ground_truth": "Claims must be submitted within 90 days."
  }'
```

### 5. Run the CI pipeline locally
```bash
python pipelines/ci_eval_pipeline.py \
  --input  tests/sample_eval_data.json \
  --output outputs/ci_eval_report.json
```

### 6. Run tests
```bash
pytest tests/ -v
```

### 7. Run with Docker
```bash
docker build -t llm-eval-framework .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... llm-eval-framework
```

---

## API Reference

### `POST /evaluate`

Runs all evaluators and returns a unified score object.

**Request body:**
```json
{
  "question":    "What is the claims submission deadline?",
  "context":     "Claims must be submitted within 90 days of the service date.",
  "answer":      "You have 90 days to submit a claim.",
  "ground_truth": "Claims must be submitted within 90 days."
}
```

**Response:**
```json
{
  "hallucination": {
    "hallucination_score": 0.04,
    "similarity": 0.96,
    "verdict": "GROUNDED",
    "is_hallucination": false,
    "threshold": 0.5,
    "mode": "sentence_level"
  },
  "output_quality": {
    "relevance": 0.91,
    "coherence": 0.88,
    "completeness": 0.85,
    "rouge_l": 0.87,
    "overall": 0.88,
    "passed": true
  },
  "ragas": {
    "faithfulness": 1.0,
    "answer_relevance": 1.0,
    "context_recall": 1.0,
    "mode": "mock",
    "available": true
  },
  "duration_seconds": 0.312,
  "evaluator_mode": "mock"
}
```

**Additional endpoints:**

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /health` | Key status + RAGAS availability |
| `POST /evaluate/hallucination` | Hallucination scores only |
| `POST /evaluate/output` | Output quality scores only |

---

## How CI Works

Every push to `main` and every pull request triggers the
[`eval-ci.yml`](.github/workflows/eval-ci.yml) workflow:
```
Push / PR to main
       │
       ▼
┌─────────────────────────────────────────────┐
│           GitHub Actions Runner             │
│                                             │
│  1. actions/checkout@v4                     │
│  2. actions/setup-python@v5  (Python 3.11)  │
│  3. pip install -r requirements.txt         │
│  4. pytest tests/ -v --tb=short             │
│       └─ fails here if any test breaks      │
│  5. python pipelines/ci_eval_pipeline.py    │
│       --input  tests/sample_eval_data.json  │
│       --output outputs/ci_eval_report.json  │
│  6. actions/upload-artifact@v4              │
│       └─ eval-report-{run_number}.json      │
│  7. Print summary to job log                │
└─────────────────────────────────────────────┘
```

**Key design decisions:**

- **No API key required in CI.** All evaluators fall back to mock mode automatically. Set `OPENAI_API_KEY` in repo **Settings → Secrets → Actions** to enable live scoring.
- **Tests are a hard gate** (`continue-on-error: false`). The pipeline only runs if pytest passes.
- **Pipeline failures are soft** (`continue-on-error: true`). The report artifact is uploaded even if scores fall below threshold, so you can inspect what regressed.
- **Artifacts retained for 30 days.** Find them under **Actions → your run → Artifacts**.

---

## Sample Report

After the CI pipeline runs, the JSON report has this structure:
```json
{
  "generated_at": "2026-03-26T10:42:00+00:00",
  "mode": "mock",
  "summary": {
    "total": 5,
    "passed": 4,
    "failed": 1,
    "pass_rate": 0.8,
    "hallucination_rate": 0.2,
    "avg_output_score": 0.84,
    "regressions": 1,
    "mode": "mock"
  },
  "cases": [
    {
      "id": "case_001",
      "question": "What is the claims submission deadline?",
      "answer": "You must submit claims within 90 days of the service date.",
      "hallucination_score": 0.04,
      "similarity": 0.96,
      "is_hallucination": false,
      "verdict": "GROUNDED",
      "output_overall": 0.91,
      "passed": true
    },
    {
      "id": "case_005",
      "question": "Is dental coverage included?",
      "answer": "Dental coverage is fully included in all plans at no extra cost.",
      "hallucination_score": 0.71,
      "similarity": 0.29,
      "is_hallucination": true,
      "verdict": "HALLUCINATED",
      "output_overall": 0.63,
      "passed": false
    }
  ]
}
```

> Download the latest report from the **Actions** tab → most recent workflow run
> → **Artifacts** section → `eval-report-{N}`.

---

## Governance & Compliance

| Requirement | How This Framework Addresses It |
|---|---|
| **EU AI Act Art. 13 — Transparency** | `hallucination_score` + `verdict` string per output provide the numeric "degree of factual grounding" required in technical documentation |
| **EU AI Act Art. 9 — Risk Management** | Regression threshold enforcement with per-case pass/fail audit trail |
| **Audit Reproducibility** | Offline/mock mode is fully deterministic; every run produces a timestamped JSON artefact |
| **Human Oversight** | CI report highlights regressions for human review; `HALLUCINATED` verdicts escalate to reviewers |
| **Scalable Validation** | Replaces line-by-line human review of N test cases with automated semantic similarity scoring |

---

## Author

**Kumar Puvvalla** — AI Engineer | Generative AI Systems | LLM Evaluation

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/kumar-puvvalla-827a95394/)
[![GitHub](https://img.shields.io/badge/GitHub-Kr1021-black?logo=github)](https://github.com/Kr1021)

---

*Proprietary scoring thresholds, internal model configurations, and enterprise
deployment details are not included in this public repository.*
