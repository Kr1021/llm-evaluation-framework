"""
evaluators/prompt_regression.py

Loads a YAML test suite of prompts + expected outputs, runs each through
an LLM (OpenAI when OPENAI_API_KEY is set, otherwise a local mock),
compares actual vs expected using TF-IDF cosine similarity, and flags
regressions where the similarity score drops below a threshold.

YAML test suite format (see tests/prompt_suite.yaml for a full example):

    suite_name: my_regression_suite
    threshold: 0.75          # optional – overrides the default 0.7
    cases:
      - id: case_001
        prompt: "What is the claims submission deadline?"
        expected: "Claims must be submitted within 90 days."
        threshold: 0.8       # optional – overrides suite-level threshold
      - id: case_002
        prompt: "Who is eligible for coverage?"
        expected: "All full-time employees are eligible."
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.70          # similarity below this → regression
DEFAULT_MODEL     = "gpt-4o-mini" # used only when OPENAI_API_KEY is set


# ---------------------------------------------------------------------------
# Mock LLM (used when OPENAI_API_KEY is not set)
# ---------------------------------------------------------------------------

class _MockLLM:
    """
    Returns the prompt back as the answer so tests are always runnable
    without an API key.  Similarity against the expected output will reflect
    how closely the test author wrote the expected text relative to the prompt.
    In practice this is used to verify the pipeline runs, not to get real scores.
    """

    def complete(self, prompt: str) -> str:
        return prompt


# ---------------------------------------------------------------------------
# OpenAI LLM wrapper
# ---------------------------------------------------------------------------

class _OpenAILLM:
    """Thin wrapper around the OpenAI chat completion API."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> None:
        from openai import OpenAI  # lazy import — only needed in live mode
        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result for a single test case."""

    id: str
    prompt: str
    expected: str
    actual: str
    similarity: float
    threshold: float
    passed: bool
    regression: bool          # True when similarity < threshold
    mode: str                 # "live" | "mock"


@dataclass
class SuiteResult:
    """Aggregated result for the full test suite."""

    suite_name: str
    total: int
    passed: int
    failed: int
    regressions: int
    pass_rate: float
    cases: List[CaseResult] = field(default_factory=list)
    mode: str = "live"

    # Convenience helpers
    @property
    def regression_ids(self) -> List[str]:
        return [c.id for c in self.cases if c.regression]

    @property
    def failed_ids(self) -> List[str]:
        return [c.id for c in self.cases if not c.passed]


# ---------------------------------------------------------------------------
# Similarity scorer
# ---------------------------------------------------------------------------

def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two strings using TF-IDF vectors.
    Returns a float in [0, 1].  No API or model download required.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer().fit([text_a, text_b])
    vectors = vectorizer.transform([text_a, text_b])
    score = float(cosine_similarity(vectors[0], vectors[1])[0][0])
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

class PromptRegressionTester:
    """
    Loads a YAML test suite, runs each prompt through an LLM, and flags
    regressions where the similarity score drops below the threshold.

    Parameters
    ----------
    threshold  : float  – default similarity threshold (overridden per-case
                          or per-suite if specified in the YAML).
    model      : str    – OpenAI model name (ignored when no API key).
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.threshold = threshold
        self.model = model
        self._llm = self._build_llm()
        self._mode = "live" if _has_api_key() else "mock"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_llm():
        if _has_api_key():
            return _OpenAILLM()
        return _MockLLM()

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if "cases" not in data:
            raise ValueError(f"YAML file '{path}' must contain a 'cases' key.")
        return data

    def _run_case(
        self,
        case: Dict,
        suite_threshold: float,
    ) -> CaseResult:
        case_id   = case.get("id", "unknown")
        prompt    = case["prompt"]
        expected  = case["expected"]
        threshold = float(case.get("threshold", suite_threshold))

        actual     = self._llm.complete(prompt)
        similarity = _tfidf_similarity(actual, expected)
        passed     = similarity >= threshold
        regression = not passed          # any failure == regression

        return CaseResult(
            id=case_id,
            prompt=prompt,
            expected=expected,
            actual=actual,
            similarity=round(similarity, 6),
            threshold=threshold,
            passed=passed,
            regression=regression,
            mode=self._mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_suite(self, yaml_path: str) -> SuiteResult:
        """
        Load the YAML test suite at *yaml_path* and run every case.

        Parameters
        ----------
        yaml_path : str – path to the YAML file.

        Returns
        -------
        SuiteResult  – aggregated pass/fail/regression report.
        """
        data           = self._load_yaml(yaml_path)
        suite_name     = data.get("suite_name", "unnamed_suite")
        suite_threshold = float(data.get("threshold", self.threshold))
        raw_cases      = data.get("cases", [])

        results: List[CaseResult] = []
        for raw in raw_cases:
            result = self._run_case(raw, suite_threshold)
            results.append(result)

        total      = len(results)
        passed     = sum(1 for r in results if r.passed)
        failed     = total - passed
        regressions = sum(1 for r in results if r.regression)
        pass_rate  = round(passed / total, 4) if total else 0.0

        return SuiteResult(
            suite_name=suite_name,
            total=total,
            passed=passed,
            failed=failed,
            regressions=regressions,
            pass_rate=pass_rate,
            cases=results,
            mode=self._mode,
        )

    def run_cases(
        self,
        cases: List[Dict],
        suite_name: str = "inline_suite",
    ) -> SuiteResult:
        """
        Run a list of case dicts directly (no YAML file needed).

        Each dict must have ``"prompt"`` and ``"expected"`` keys.
        ``"id"`` and ``"threshold"`` are optional.

        Parameters
        ----------
        cases      : list of dicts
        suite_name : str – label for the returned SuiteResult

        Returns
        -------
        SuiteResult
        """
        results: List[CaseResult] = []
        for raw in cases:
            result = self._run_case(raw, self.threshold)
            results.append(result)

        total       = len(results)
        passed      = sum(1 for r in results if r.passed)
        failed      = total - passed
        regressions = sum(1 for r in results if r.regression)
        pass_rate   = round(passed / total, 4) if total else 0.0

        return SuiteResult(
            suite_name=suite_name,
            total=total,
            passed=passed,
            failed=failed,
            regressions=regressions,
            pass_rate=pass_rate,
            cases=results,
            mode=self._mode,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_regression_suite(
    yaml_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_MODEL,
) -> SuiteResult:
    """
    One-shot convenience wrapper.  Load *yaml_path* and run the full suite.

    Parameters
    ----------
    yaml_path : str   – path to the YAML test suite file.
    threshold : float – default similarity threshold (default 0.70).
    model     : str   – OpenAI model name (ignored when no API key is set).

    Returns
    -------
    SuiteResult
    """
    return PromptRegressionTester(
        threshold=threshold, model=model
    ).run_suite(yaml_path)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())
