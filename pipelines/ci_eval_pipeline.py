"""
pipelines/ci_eval_pipeline.py

CLI-runnable evaluation pipeline designed for GitHub Actions CI.
Loads sample data (JSON), runs every evaluator in mock-safe mode,
and writes a structured JSON report to disk.

Usage:
    python pipelines/ci_eval_pipeline.py \
        --input  tests/sample_eval_data.json \
        --output outputs/ci_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from evaluators.hallucination import score_hallucination_sentences
from evaluators.output_scorer import OutputScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _load_input(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Input file must be a JSON array of records, got: {type(data)}")
    return data


def _run_record(record: Dict, scorer: OutputScorer) -> Dict:
    """Evaluate a single record and return a result dict."""
    question     = record["question"]
    context      = record["context"]
    answer       = record["answer"]
    ground_truth = record.get("ground_truth")

    # --- Hallucination ---
    h = score_hallucination_sentences(answer=answer, context=context)

    # --- Output quality ---
    s = scorer.score(query=question, response=answer, reference=ground_truth)

    passed = s.passed and not h.is_hallucination

    return {
        "id":                   record.get("id", "unknown"),
        "question":             question,
        "answer":               answer,
        # hallucination
        "hallucination_score":  h.hallucination_score,
        "similarity":           h.similarity,
        "is_hallucination":     h.is_hallucination,
        "verdict":              h.verdict,
        # output quality
        "output_scores":        s.scores,
        "output_overall":       s.overall,
        "output_passed":        s.passed,
        # combined
        "passed":               passed,
    }


def _build_summary(results: List[Dict], mode: str) -> Dict:
    total      = len(results)
    passed     = sum(1 for r in results if r["passed"])
    h_flags    = sum(1 for r in results if r["is_hallucination"])
    avg_score  = sum(r["output_overall"] for r in results) / total if total else 0.0
    regressions = total - passed

    return {
        "total":              total,
        "passed":             passed,
        "failed":             regressions,
        "pass_rate":          round(passed / total, 4) if total else 0.0,
        "hallucination_rate": round(h_flags / total, 4) if total else 0.0,
        "avg_output_score":   round(avg_score, 4),
        "regressions":        regressions,
        "mode":               mode,
    }


def run_ci_pipeline(input_path: str, output_path: str) -> Dict:
    """
    Main entry point.  Loads *input_path*, runs all evaluators, writes
    the JSON report to *output_path*, and returns the report dict.
    """
    mode = "live" if _has_api_key() else "mock"
    logger.info(f"Starting CI pipeline | mode={mode} | input={input_path}")

    records = _load_input(input_path)
    logger.info(f"Loaded {len(records)} records")

    scorer  = OutputScorer()
    results = []

    for i, record in enumerate(records):
        logger.info(f"Evaluating record {i + 1}/{len(records)} id={record.get('id', i)}")
        try:
            result = _run_record(record, scorer)
        except Exception as exc:
            logger.warning(f"Record {record.get('id', i)} failed: {exc}")
            result = {
                "id":    record.get("id", str(i)),
                "error": str(exc),
                "passed": False,
            }
        results.append(result)

    summary = _build_summary(results, mode)
    logger.info(
        f"Pipeline complete — pass_rate={summary['pass_rate']} "
        f"hallucination_rate={summary['hallucination_rate']} "
        f"mode={mode}"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode":         mode,
        "summary":      summary,
        "cases":        results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"Report written to {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI Evaluation Pipeline")
    parser.add_argument(
        "--input",
        default="tests/sample_eval_data.json",
        help="Path to the JSON input file (array of records).",
    )
    parser.add_argument(
        "--output",
        default="outputs/ci_eval_report.json",
        help="Path to write the JSON report.",
    )
    args = parser.parse_args()
    run_ci_pipeline(args.input, args.output)
