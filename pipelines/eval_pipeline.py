import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd

from evaluators.hallucination_detector import HallucinationDetector
from evaluators.output_scorer import OutputScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
      model_name: str = "gpt-4"
      hallucination_threshold: float = 0.75
      scoring_threshold: float = 0.70
      s3_bucket: Optional[str] = None
      s3_prefix: str = "eval-results"
      save_locally: bool = True
      output_dir: str = "outputs"


@dataclass
class PipelineResult:
      total: int
      hallucination_rate: float
      avg_output_score: float
      pass_rate: float
      failed_records: List[dict]


class EvalPipeline:
      def __init__(self, config: PipelineConfig):
                self.config = config
                self.detector = HallucinationDetector(
                    model_name=config.model_name,
                    threshold=config.hallucination_threshold,
                )
                self.scorer = OutputScorer(
                    model_name=config.model_name,
                    pass_threshold=config.scoring_threshold,
                )
                self._s3 = boto3.client("s3") if config.s3_bucket else None

      def _load_records(self, source: str) -> List[dict]:
                path = Path(source)
                if path.suffix == ".json":
                              with open(path) as f:
                                                return json.load(f)
                elif path.suffix == ".csv":
                              return pd.read_csv(path).to_dict(orient="records")
                          raise ValueError(f"Unsupported file format: {path.suffix}")

      def _save_results(self, results: List[dict], run_id: str) -> None:
                df = pd.DataFrame(results)
                if self.config.save_locally:
                              out_path = Path(self.config.output_dir) / f"{run_id}.csv"
                              out_path.parent.mkdir(parents=True, exist_ok=True)
                              df.to_csv(out_path, index=False)
                              logger.info(f"Results saved to {out_path}")
                          if self._s3:
                                        key = f"{self.config.s3_prefix}/{run_id}.csv"
                                        self._s3.put_object(
                                            Bucket=self.config.s3_bucket,
                                            Key=key,
                                            Body=df.to_csv(index=False),
                                        )
                                        logger.info(f"Results uploaded to s3://{self.config.s3_bucket}/{key}")

            def run(self, source: str, run_id: str = "eval_run") -> PipelineResult:
                      records = self._load_records(source)
                      logger.info(f"Loaded {len(records)} records from {source}")

        hallucination_results = self.detector.batch_detect(records)
        scoring_results = self.scorer.batch_score(records)

        combined = []
        failed = []
        hallucination_flags = 0

        for i, record in enumerate(records):
                      h = hallucination_results[i]
                      s = scoring_results[i]

            entry = {
                              **record,
                              "hallucination_score": h.score,
                              "is_hallucination": h.is_hallucination,
                              "flagged_spans": h.flagged_spans,
                              "output_score": s.overall,
                              "passed": s.passed and not h.is_hallucination,
            }
            combined.append(entry)

            if h.is_hallucination:
                              hallucination_flags += 1
                          if not entry["passed"]:
                                            failed.append(entry)

        self._save_results(combined, run_id)

        return PipelineResult(
                      total=len(records),
                      hallucination_rate=hallucination_flags / len(records),
                      avg_output_score=sum(s.overall for s in scoring_results) / len(scoring_results),
                      pass_rate=(len(records) - len(failed)) / len(records),
                      failed_records=failed,
        )


def run_from_config(config_path: str, source: str, run_id: str = "eval_run") -> PipelineResult:
      with open(config_path) as f:
                cfg = PipelineConfig(**json.load(f))
            pipeline = EvalPipeline(cfg)
    return pipeline.run(source, run_id)
