from dataclasses import dataclass, field
from typing import List, Optional
import re

import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


@dataclass
class HallucinationResult:
      query: str
      response: str
      context: str
      score: float
      is_hallucination: bool
      flagged_spans: List[str] = field(default_factory=list)
      confidence: float = 0.0


class HallucinationDetector:
      def __init__(
                self,
                model_name: str = "gpt-4",
                embedding_model: str = "all-MiniLM-L6-v2",
                threshold: float = 0.75,
                temperature: float = 0.0,
      ):
                self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
                self.embedder = SentenceTransformer(embedding_model)
                self.threshold = threshold

      def _semantic_similarity(self, text_a: str, text_b: str) -> float:
                embeddings = self.embedder.encode([text_a, text_b], convert_to_tensor=True)
                return float(util.cos_sim(embeddings[0], embeddings[1]))

      def _extract_claims(self, response: str) -> List[str]:
                prompt = (
                              f"Extract all factual claims from the following response as a numbered list.\n\n"
                              f"Response:\n{response}\n\nClaims:"
                )
                result = self.llm([HumanMessage(content=prompt)])
                lines = result.content.strip().split("\n")
                claims = [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines if line.strip()]
                return claims

      def _verify_claim_against_context(self, claim: str, context: str) -> float:
                prompt = (
                              f"Given the following context, rate how well-supported the claim is on a scale from 0.0 to 1.0.\n\n"
                              f"Context:\n{context}\n\nClaim:\n{claim}\n\n"
                              f"Return only a float between 0.0 and 1.0."
                )
                result = self.llm([HumanMessage(content=prompt)])
                try:
                              return float(result.content.strip())
except ValueError:
            return 0.0

    def detect(self, query: str, response: str, context: str) -> HallucinationResult:
              claims = self._extract_claims(response)
              if not claims:
                            return HallucinationResult(
                                              query=query,
                                              response=response,
                                              context=context,
                                              score=1.0,
                                              is_hallucination=False,
                                              confidence=0.5,
                            )

              scores = [self._verify_claim_against_context(c, context) for c in claims]
              avg_score = float(np.mean(scores))
              flagged = [claims[i] for i, s in enumerate(scores) if s < self.threshold]

        return HallucinationResult(
                      query=query,
                      response=response,
                      context=context,
                      score=avg_score,
                      is_hallucination=avg_score < self.threshold,
                      flagged_spans=flagged,
                      confidence=float(np.std(scores)),
        )

    def batch_detect(
              self, records: List[dict]
    ) -> List[HallucinationResult]:
              return [
                            self.detect(r["query"], r["response"], r["context"])
                            for r in records
              ]
