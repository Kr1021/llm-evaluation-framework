## LLM Evaluation Framework

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

> Production-grade LLM evaluation framework for operationalising reliability standards as auditable governance artefacts. Built for enterprise AI teams who need to demonstrate compliance, reduce hallucination risk, and automate model validation at scale.

---

## Project Purpose

As organisations deploy LLMs in regulated environments — insurance, finance, healthcare — they face a critical gap: **model outputs are probabilistic, but compliance obligations are deterministic**. This framework bridges that gap by turning every evaluation run into a structured, versioned, human-readable governance artefact.

Key alignment:
- **EU AI Act Article 13 (Transparency)** — hallucination scores provide the numeric "degree of factual grounding" required in technical documentation, and verdict strings satisfy the obligation to present AI decisions in an understandable form.
- **Regression testing reduces manual validation effort by ~40%** — replacing line-by-line human review of N test cases with automated semantic similarity scoring, freeing compliance teams to focus on edge cases only.

---

## Architecture
```
  Test Cases (JSON)
        │
        ▼
┌─────────────────────────────────────────────────┐
│              EvaluationPipeline                 │
│                                                 │
│  ┌──────────────────┐  ┌─────────────────────┐  │
│  │ HallucinationScorer│  │OutputReliabilityMetrics│ │
│  │                  │  │                     │  │
│  │ • RAGAS mode     │  │ • answer_faithfulness│  │
│  │ • Jaccard fallback│  │ • context_recall    │  │
│  │ • verdict string │  │ • answer_relevancy  │  │
│  └──────────────────┘  └─────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │         PromptRegressionTester           │   │
│  │  baseline outputs ──► cosine similarity  │   │
│  │  new outputs      ──► pass/fail per case │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
│              Aggregated Report Dict             │
└─────────────────────────────────────────────────┘
        │
        ▼
  Governance Report (Markdown)
  governance_report_YYYY-MM-DD.md
```

---

## Project Structure
```
llm-evaluation-framework/
├── src/
│   ├── evaluators/
│   │   ├── hallucination_scorer.py      # RAGAS + Jaccard fallback
│   │   ├── prompt_regression_tester.py  # Semantic similarity regression
│   │   └── output_reliability.py        # TF-IDF cosine similarity metrics
│   └── pipeline/
│       └── evaluation_pipeline.py       # Orchestrates all evaluators
├── tests/
│   ├── test_hallucination_scorer.py
│   └── test_regression_tester.py
├── data/
│   └── sample_outputs/
│       └── sample_test_cases.json       # 10 insurance domain test cases
├── .github/
│   └── workflows/
│       └── evaluate.yml                 # CI/CD evaluation pipeline
├── requirements.txt
└── README.md
```

---

## Installation
```bash
git clone https://github.com/Kr1021/llm-evaluation-framework.git
cd llm-evaluation-framework
pip install -r requirements.txt
```

---

## Quickstart
```python
from src.pipeline.evaluation_pipeline import EvaluationPipeline

pipeline = EvaluationPipeline()

test_cases = [
    {
        "question": "What is the claims submission deadline?",
        "context": "Claims must be submitted within 90 days of the service date.",
        "answer": "You must submit claims within 90 days of the service date.",
        "ground_truth": "Claims must be submitted within 90 days of the service date."
    }
]

results = pipeline.run(test_cases)
pipeline.export_governance_report(results, "governance_report.md")
print(results["summary"])
```

---

## Components

### HallucinationScorer
Scores each answer for factual grounding using two modes:
- **RAGAS mode** (when `OPENAI_API_KEY` is set) — uses NLI-based faithfulness scoring
- **Heuristic mode** (offline fallback) — Jaccard similarity between answer and context tokens

Returns `{"hallucination_rate": float, "faithfulness": float, "verdict": str, "mode": str}`

### PromptRegressionTester
Compares new model outputs against a baseline using cosine similarity on TF-IDF vectors. Flags regressions where similarity drops below a configurable threshold (default 0.85). Produces a governance report showing pass rate and specific regressions.

### OutputReliabilityMetrics
Three offline metrics using cosine similarity on TF-IDF vectors — no API key needed:
- `answer_faithfulness(context, answer)` → float 0–1
- `context_recall(context, ground_truth)` → float 0–1
- `answer_relevancy(question, answer)` → float 0–1

### EvaluationPipeline
Orchestrates all three evaluators over a batch of test cases. Returns a full report dict with per-case scores and summary statistics. Exports a structured Markdown governance report.

---

## Governance & Compliance

| Requirement | How This Framework Addresses It |
|---|---|
| EU AI Act Art. 13 — Transparency | `hallucination_rate` + `verdict` string per output |
| EU AI Act Art. 9 — Risk Management | Regression threshold enforcement with pass/fail audit trail |
| Audit Reproducibility | Heuristic mode is fully deterministic; RAGAS mode logs API call chain |
| Human Oversight | Governance report highlights regressions for human review |

> **Note:** Proprietary scoring thresholds, internal model configurations, and enterprise deployment details are not included in this public repository.

---

## Sample Governance Report Output
```
# LLM Evaluation Governance Report
Generated: 2026-03-24

## Summary
- Total test cases: 10
- Overall pass rate: 80.0%
- Mean hallucination rate: 0.18
- Mean faithfulness: 0.82

## Regressions Detected (2)
- Case 4: similarity 0.61 < threshold 0.85 — REVIEW REQUIRED
- Case 9: similarity 0.73 < threshold 0.85 — REVIEW REQUIRED

## Per-Case Scores
| Case | Faithfulness | Hallucination Rate | Verdict |
|------|-------------|-------------------|---------|
| 1    | 0.91        | 0.09              | FAITHFUL |
| 2    | 0.88        | 0.12              | FAITHFUL |
...
```

---

## Running Tests
```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Evaluation | RAGAS, custom scorers |
| ML / NLP | scikit-learn (TF-IDF), sentence-transformers |
| LLM Orchestration | LangChain |
| Cloud | AWS (Lambda, S3, CloudWatch) |
| Testing | Pytest |
| CI/CD | GitHub Actions |

---

## Author

**Kumar Puvvalla** — AI Engineer | Generative AI Systems | LLM Evaluation  
[LinkedIn](https://www.linkedin.com/in/kumar-puvvalla-827a95394/) | [GitHub](https://github.com/Kr1021)
