"""
evaluators/hallucination.py

Computes a hallucination score for a (answer, context) pair using
sentence-transformers cosine similarity — no external API required.

Hallucination score  = 1 - semantic_similarity(answer, context)

A score close to 0 means the answer is well-grounded in the context.
A score close to 1 means the answer is likely hallucinated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_HALLUCINATION_THRESHOLD = 0.5


@dataclass
class HallucinationScore:
    answer: str
    context: str
    similarity: float
    hallucination_score: float
    is_hallucination: bool
    threshold: float
    verdict: str
    sentence_scores: List[Dict] = field(default_factory=list)
    model: str = DEFAULT_MODEL


class HallucinationScorer:

    def __init__(                                   # ← fix 1: was "def init("
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: List[str]):
        return self._model.encode(texts, convert_to_tensor=True)

    def _cosine(self, text_a: str, text_b: str) -> float:   # ← fix 2: was "_cosine(self, texta:"
        embeddings = self._encode([text_a, text_b])
        score = float(util.cos_sim(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _split_sentences(text: str) -> List[str]:           # ← fix 3: was "splitsentences("
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _hallucination_from_similarity(self, similarity: float) -> float:  # ← fix 4
        return round(1.0 - similarity, 6)

    def _make_verdict(self, hallucination_score: float) -> str:            # ← fix 4
        return "HALLUCINATED" if hallucination_score >= self.threshold else "GROUNDED"

    def score(self, answer: str, context: str) -> HallucinationScore:
        similarity = self._cosine(answer, context)
        h_score = self._hallucination_from_similarity(similarity)
        verdict = self._make_verdict(h_score)
        return HallucinationScore(
            answer=answer,
            context=context,
            similarity=similarity,
            hallucination_score=h_score,
            is_hallucination=(h_score >= self.threshold),
            threshold=self.threshold,
            verdict=verdict,
            model=self.model_name,
        )

    def score_sentences(self, answer: str, context: str) -> HallucinationScore:
        sentences = self._split_sentences(answer)
        if not sentences:
            return HallucinationScore(
                answer=answer, context=context, similarity=1.0,
                hallucination_score=0.0, is_hallucination=False,
                threshold=self.threshold, verdict="GROUNDED",
                model=self.model_name,
            )
        sentence_scores: List[Dict] = []
        for sentence in sentences:
            sim = self._cosine(sentence, context)
            h = self._hallucination_from_similarity(sim)
            sentence_scores.append({
                "sentence": sentence,
                "similarity": sim,
                "hallucination_score": h,
                "is_hallucination": h >= self.threshold,
            })
        mean_h_score = float(np.mean([s["hallucination_score"] for s in sentence_scores]))
        mean_similarity = round(1.0 - mean_h_score, 6)
        verdict = self._make_verdict(mean_h_score)
        return HallucinationScore(
            answer=answer, context=context,
            similarity=mean_similarity, hallucination_score=mean_h_score,
            is_hallucination=(mean_h_score >= self.threshold),
            threshold=self.threshold, verdict=verdict,
            sentence_scores=sentence_scores, model=self.model_name,
        )

    def batch_score(
        self, records: List[Dict], sentence_level: bool = False
    ) -> List[HallucinationScore]:
        fn = self.score_sentences if sentence_level else self.score
        return [fn(r["answer"], r["context"]) for r in records]


def score_hallucination(
    answer: str,
    context: str,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
) -> HallucinationScore:
    return HallucinationScorer(model_name=model_name, threshold=threshold).score(
        answer, context
    )


def score_hallucination_sentences(
    answer: str,
    context: str,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
) -> HallucinationScore:
    return HallucinationScorer(
        model_name=model_name, threshold=threshold
    ).score_sentences(answer, context)


def batch_score_hallucination(
    records: List[Dict],
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    sentence_level: bool = False,
) -> List[HallucinationScore]:
    return HallucinationScorer(
        model_name=model_name, threshold=threshold
    ).batch_score(records, sentence_level=sentence_level)
