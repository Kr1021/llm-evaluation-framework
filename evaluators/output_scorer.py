from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


@dataclass
class ScoringResult:
      query: str
      response: str
      reference: Optional[str]
      scores: Dict[str, float]
      overall: float
      passed: bool


class OutputScorer:
      def __init__(
                self,
                model_name: str = "gpt-4",
                embedding_model: str = "all-MiniLM-L6-v2",
                pass_threshold: float = 0.7,
      ):
                self.llm = ChatOpenAI(model_name=model_name, temperature=0.0)
                self.embedder = SentenceTransformer(embedding_model)
                self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                self.pass_threshold = pass_threshold

      def _coherence_score(self, response: str) -> float:
                prompt = (
                              "Rate the coherence and fluency of the following text on a scale from 0.0 to 1.0. "
                              "Return only the float.\n\nText:\n" + response
                )
                result = self.llm([HumanMessage(content=prompt)])
                try:
                              return float(result.content.strip())
except ValueError:
            return 0.0

    def _relevance_score(self, query: str, response: str) -> float:
              embeddings = self.embedder.encode([query, response], convert_to_tensor=True)
              return float(util.cos_sim(embeddings[0], embeddings[1]))

    def _rouge_score(self, response: str, reference: str) -> float:
              scores = self.rouge.score(reference, response)
              return scores["rougeL"].fmeasure

    def _completeness_score(self, query: str, response: str) -> float:
              prompt = (
                            f"Given the query below, rate how completely the response addresses it (0.0 to 1.0). "
                            f"Return only the float.\n\nQuery: {query}\n\nResponse: {response}"
              )
              result = self.llm([HumanMessage(content=prompt)])
              try:
                            return float(result.content.strip())
except ValueError:
            return 0.0

    def score(
              self,
              query: str,
              response: str,
              reference: Optional[str] = None,
    ) -> ScoringResult:
              scores: Dict[str, float] = {}
              scores["coherence"] = self._coherence_score(response)
              scores["relevance"] = self._relevance_score(query, response)
              scores["completeness"] = self._completeness_score(query, response)

        if reference:
                      scores["rouge_l"] = self._rouge_score(response, reference)

        overall = float(np.mean(list(scores.values())))

        return ScoringResult(
                      query=query,
                      response=response,
                      reference=reference,
                      scores=scores,
                      overall=overall,
                      passed=overall >= self.pass_threshold,
        )

    def batch_score(self, records: List[dict]) -> List[ScoringResult]:
              return [
                            self.score(
                                              r["query"],
                                              r["response"],
                                              r.get("reference"),
                            )
                            for r in records
              ]
