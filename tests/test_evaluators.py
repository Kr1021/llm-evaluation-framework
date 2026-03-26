"""
tests/test_evaluators.py

Pytest suite for all evaluator modules.
All tests run fully offline — no OPENAI_API_KEY required.
"""

import pytest
from evaluators.hallucination import (
    HallucinationScorer,
    score_hallucination,
    score_hallucination_sentences,
)
from evaluators.output_scorer import OutputScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GROUNDED_ANSWER  = "Claims must be submitted within 90 days of the service date."
CONTEXT          = "Claims must be submitted within 90 days of the service date."
HALLUCINATED_ANS = "Claims must be submitted within 7 days and require a notarised form."
QUESTION         = "What is the claims submission deadline?"


# ---------------------------------------------------------------------------
# Hallucination scorer tests
# ---------------------------------------------------------------------------

class TestHallucinationScorer:

    def test_grounded_answer_has_low_hallucination_score(self):
        result = score_hallucination(answer=GROUNDED_ANSWER, context=CONTEXT)
        assert result.hallucination_score < 0.5
        assert result.verdict == "GROUNDED"
        assert result.is_hallucination is False

    def test_hallucinated_answer_has_high_hallucination_score(self):
        result = score_hallucination(answer=HALLUCINATED_ANS, context=CONTEXT)
        # score should be higher than the grounded answer
        grounded = score_hallucination(answer=GROUNDED_ANSWER, context=CONTEXT)
        assert result.hallucination_score > grounded.hallucination_score

    def test_similarity_is_between_0_and_1(self):
        result = score_hallucination(answer=GROUNDED_ANSWER, context=CONTEXT)
        assert 0.0 <= result.similarity <= 1.0

    def test_hallucination_score_plus_similarity_equals_1(self):
        result = score_hallucination(answer=GROUNDED_ANSWER, context=CONTEXT)
        assert abs(result.hallucination_score + result.similarity - 1.0) < 1e-4

    def test_custom_threshold(self):
        result = score_hallucination(
            answer=GROUNDED_ANSWER, context=CONTEXT, threshold=0.01
        )
        # With a tiny threshold everything should be flagged
        assert result.is_hallucination is True
        assert result.verdict == "HALLUCINATED"

    def test_sentence_level_returns_sentence_scores(self):
        result = score_hallucination_sentences(
            answer="Claims are due in 90 days. You can file online.", context=CONTEXT
        )
        assert isinstance(result.sentence_scores, list)
        assert len(result.sentence_scores) == 2
        for s in result.sentence_scores:
            assert "sentence" in s
            assert "hallucination_score" in s
            assert "similarity" in s

    def test_empty_answer_returns_grounded(self):
        result = score_hallucination_sentences(answer="", context=CONTEXT)
        assert result.hallucination_score == 0.0
        assert result.verdict == "GROUNDED"

    def test_batch_score(self):
        scorer = HallucinationScorer()
        records = [
            {"answer": GROUNDED_ANSWER,  "context": CONTEXT},
            {"answer": HALLUCINATED_ANS, "context": CONTEXT},
        ]
        results = scorer.batch_score(records)
        assert len(results) == 2
        assert results[0].hallucination_score < results[1].hallucination_score


# ---------------------------------------------------------------------------
# Output scorer tests
# ---------------------------------------------------------------------------

class TestOutputScorer:

    def test_score_returns_result_with_expected_keys(self):
        scorer = OutputScorer()
        result = scorer.score(query=QUESTION, response=GROUNDED_ANSWER)
        assert "relevance" in result.scores
        assert 0.0 <= result.overall <= 1.0
        assert isinstance(result.passed, bool)

    def test_relevant_answer_scores_higher_than_irrelevant(self):
        scorer  = OutputScorer()
        good    = scorer.score(query=QUESTION, response=GROUNDED_ANSWER)
        bad     = scorer.score(query=QUESTION, response="The weather in Paris is sunny.")
        assert good.scores["relevance"] > bad.scores["relevance"]

    def test_rouge_l_computed_when_reference_provided(self):
        scorer = OutputScorer()
        result = scorer.score(
            query=QUESTION,
            response=GROUNDED_ANSWER,
            reference=CONTEXT,
        )
        assert "rouge_l" in result.scores
        assert result.scores["rouge_l"] > 0.0

    def test_rouge_l_absent_without_reference(self):
        scorer = OutputScorer()
        result = scorer.score(query=QUESTION, response=GROUNDED_ANSWER)
        assert "rouge_l" not in result.scores

    def test_batch_score_length_matches_input(self):
        scorer  = OutputScorer()
        records = [
            {"query": QUESTION, "response": GROUNDED_ANSWER},
            {"query": QUESTION, "response": HALLUCINATED_ANS},
        ]
        results = scorer.batch_score(records)
        assert len(results) == 2
```
