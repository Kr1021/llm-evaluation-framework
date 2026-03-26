"""
evaluators/ragas_scorer.py

Functions to score a (question, context, answer) triple using the RAGAS library.
Returns faithfulness, answer_relevance, and context_recall scores.

Falls back to a deterministic MockLLM / MockEmbeddings implementation when no
OPENAI_API_KEY is found in the environment, so the module is always importable
and testable without any external API credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# RAGAS imports
# ---------------------------------------------------------------------------
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextRecall,
    Faithfulness,
)

# ---------------------------------------------------------------------------
# LangChain imports – used to wire up the mock when no key is present
# ---------------------------------------------------------------------------
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Mock LLM & Embeddings (used when OPENAI_API_KEY is not set)
# ---------------------------------------------------------------------------

class _MockLLM(BaseChatModel):
    """Deterministic stub that always returns '1' so every metric scores 1.0
    in offline / CI environments."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="1"))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        return self._generate(messages, stop=stop)


class _MockEmbeddings(Embeddings):
    """Returns identical unit vectors so cosine similarity is always 1.0."""

    _DIM: int = 8

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        unit = [1.0 / self._DIM ** 0.5] * self._DIM
        return [unit for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [1.0 / self._DIM ** 0.5] * self._DIM


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGASResult:
    """Container for the three RAGAS scores returned by this module."""

    question: str
    context: str
    answer: str
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    ground_truth: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    mode: str = "live"  # "live" | "mock"

    def __post_init__(self) -> None:
        self.scores = {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_recall": self.context_recall,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    """Return True if a non-empty OPENAI_API_KEY is present in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _build_ragas_llm_and_embeddings():
    """
    Return (llm, embeddings) suitable for RAGAS.

    - OPENAI_API_KEY set  →  real ChatOpenAI + OpenAIEmbeddings
    - No key              →  _MockLLM + _MockEmbeddings
    """
    if _has_api_key():
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        llm = _MockLLM()
        embeddings = _MockEmbeddings()

    return llm, embeddings


def _make_dataset(
    question: str,
    context: str,
    answer: str,
    ground_truth: Optional[str] = None,
) -> Dataset:
    """Wrap inputs into the HuggingFace Dataset format expected by RAGAS."""
    row: Dict[str, list] = {
        "question": [question],
        "contexts": [[context]],   # RAGAS expects a list of context strings
        "answer": [answer],
    }
    if ground_truth is not None:
        row["ground_truth"] = [ground_truth]
    return Dataset.from_dict(row)


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def score_faithfulness(
    question: str,
    context: str,
    answer: str,
) -> float:
    """
    Return the RAGAS *faithfulness* score for a single (question, context, answer)
    triple.

    Faithfulness measures whether every claim in *answer* can be inferred from
    *context*.  Score is in [0, 1]; higher is better.

    Parameters
    ----------
    question : str  – the user question posed to the RAG system.
    context  : str  – the retrieved context provided to the LLM.
    answer   : str  – the generated answer to be evaluated.

    Returns
    -------
    float – faithfulness score in [0, 1].
    """
    llm, _ = _build_ragas_llm_and_embeddings()
    dataset = _make_dataset(question, context, answer)
    result = evaluate(dataset, metrics=[Faithfulness(llm=llm)])
    return float(result["faithfulness"])


def score_answer_relevance(
    question: str,
    context: str,
    answer: str,
) -> float:
    """
    Return the RAGAS *answer_relevancy* score.

    Answer relevance measures how well the *answer* addresses the *question*,
    regardless of factual grounding.  Score is in [0, 1].

    Parameters
    ----------
    question : str  – the user question.
    context  : str  – the retrieved context.
    answer   : str  – the generated answer to be evaluated.

    Returns
    -------
    float – answer relevance score in [0, 1].
    """
    llm, embeddings = _build_ragas_llm_and_embeddings()
    dataset = _make_dataset(question, context, answer)
    result = evaluate(
        dataset,
        metrics=[AnswerRelevancy(llm=llm, embeddings=embeddings)],
    )
    return float(result["answer_relevancy"])


def score_context_recall(
    question: str,
    context: str,
    answer: str,
    ground_truth: str = "",
) -> float:
    """
    Return the RAGAS *context_recall* score.

    Context recall measures how much of the information in *ground_truth* can
    be attributed to the retrieved *context*.  Score is in [0, 1].

    Parameters
    ----------
    question     : str  – the user question.
    context      : str  – the retrieved context to be evaluated.
    answer       : str  – the generated answer.
    ground_truth : str  – reference answer; falls back to *answer* when omitted.

    Returns
    -------
    float – context recall score in [0, 1].
    """
    llm, _ = _build_ragas_llm_and_embeddings()
    gt = ground_truth.strip() or answer
    dataset = _make_dataset(question, context, answer, ground_truth=gt)
    result = evaluate(dataset, metrics=[ContextRecall(llm=llm)])
    return float(result["context_recall"])


def score_all(
    question: str,
    context: str,
    answer: str,
    ground_truth: Optional[str] = None,
) -> RAGASResult:
    """
    Run all three RAGAS metrics in a single pass and return a RAGASResult.

    More efficient than calling the three individual functions separately
    because it builds the LLM/embedding objects once and runs one
    ``eva
