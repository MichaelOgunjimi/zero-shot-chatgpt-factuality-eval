"""
SOTA Baseline Implementations for ChatGPT Factuality Evaluation
==============================================================

This module implements state-of-the-art factuality evaluation metrics
specifically adapted for comparison with ChatGPT's performance on the
three core factuality tasks.

The implementations focus on providing fair, task-specific comparisons
rather than general-purpose metric calculations, aligning with the
thesis goal of evaluating ChatGPT's zero-shot factuality assessment.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import json
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import bert_score
import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy import stats
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as transformers_logging,
)

from ..data.loaders import FactualityExample
from ..utils.config import get_config
from ..utils.logging import get_logger

# Suppress transformer warnings for cleaner output
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)


@dataclass
class BaselineResult:
    """
    Standardized result format for baseline evaluations.

    This ensures consistent comparison between different SOTA methods
    and ChatGPT results across all three factuality tasks.
    """

    baseline_name: str
    task_name: str
    example_id: str
    prediction: Union[int, float, List[int]]  # Task-specific format
    confidence: float
    raw_scores: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return asdict(self)

    def get_task_specific_prediction(self) -> Union[int, float, List[int]]:
        """
        Get prediction in the format expected for the specific task.

        Returns:
            - Entailment Inference: int (0/1)
            - Summary Ranking: List[int] (ranked indices)
            - Consistency Rating: float (0-100 scale)
        """
        return self.prediction

    def is_compatible_with_task(self, task_name: str) -> bool:
        """Check if result is compatible with specified task."""
        return self.task_name == task_name


class SOTABaseline(ABC):
    """
    Abstract base class for SOTA baseline implementations.

    Defines the common interface that all baseline methods must implement
    to ensure consistent evaluation and comparison with ChatGPT.
    """

    def __init__(self, config: Optional[Dict] = None, device: Optional[str] = None):
        """
        Initialize baseline method.

        Args:
            config: Configuration dictionary
            device: Computing device (auto, cpu, cuda, mps)
        """
        self.config = config or get_config()
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False

        # Initialize logging for this baseline
        self.logger = get_logger(f"{self.__class__.__name__}")

    def _setup_device(self, device: Optional[str]) -> str:
        """Setup computing device with fallback options."""
        if device and device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @abstractmethod
    def initialize(self) -> None:
        """Initialize models and resources. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def supports_task(self, task_name: str) -> bool:
        """Check if baseline supports the specified task."""
        pass

    @abstractmethod
    def evaluate_entailment_inference(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """
        Evaluate entailment inference task (binary classification).

        Args:
            source: Source document text
            summary: Summary text to evaluate
            example_id: Unique identifier for the example

        Returns:
            BaselineResult with binary prediction (0/1)
        """
        pass

    @abstractmethod
    def evaluate_summary_ranking(
        self, source: str, summaries: List[str], example_id: str = None
    ) -> BaselineResult:
        """
        Evaluate summary ranking task.

        Args:
            source: Source document text
            summaries: List of summaries to rank
            example_id: Unique identifier for the example

        Returns:
            BaselineResult with ranked indices
        """
        pass

    @abstractmethod
    def evaluate_consistency_rating(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """
        Evaluate consistency rating task (0-100 scale).

        Args:
            source: Source document text
            summary: Summary text to evaluate
            example_id: Unique identifier for the example

        Returns:
            BaselineResult with 0-100 rating
        """
        pass

    def evaluate_example(
        self, example: FactualityExample, task_name: str
    ) -> BaselineResult:
        """
        Evaluate a single example for the specified task.

        Args:
            example: Data example to evaluate
            task_name: Name of the factuality task

        Returns:
            BaselineResult for the specified task

        Raises:
            ValueError: If task is not supported
        """
        if not self.supports_task(task_name):
            raise ValueError(
                f"{self.__class__.__name__} does not support task: {task_name}"
            )

        if not self._is_initialized:
            self.initialize()

        # Route to appropriate task-specific method
        if task_name == "entailment_inference":
            return self.evaluate_entailment_inference(
                example.source, example.summary, example.example_id
            )
        elif task_name == "summary_ranking":
            # For ranking, we need multiple summaries
            summaries = getattr(example, "summaries", [example.summary])
            return self.evaluate_summary_ranking(
                example.source, summaries, example.example_id
            )
        elif task_name == "consistency_rating":
            return self.evaluate_consistency_rating(
                example.source, example.summary, example.example_id
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")


class FactCCBaseline(SOTABaseline):
    """
    FactCC baseline implementation for factuality evaluation.

    FactCC uses a BERT-based classifier trained on synthetic data to predict
    whether a summary is factually consistent with its source document.

    Adapted for ChatGPT comparison on entailment inference and consistency rating.
    """

    def __init__(self, config: Optional[Dict] = None, device: Optional[str] = None):
        super().__init__(config, device)

        # FactCC-specific configuration
        factcc_config = self.config.get("baselines", {}).get("factcc", {})
        self.model_name = factcc_config.get(
            "model_name", "manueldeprada/FactCC"
        )
        self.batch_size = factcc_config.get("batch_size", 8)
        self.max_length = factcc_config.get("max_length", 512)
        self.cache_predictions = factcc_config.get("cache_predictions", True)

    def initialize(self) -> None:
        """Initialize FactCC model and tokenizer."""
        try:
            self.logger.info(f"Initializing FactCC model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

            self._is_initialized = True
            self.logger.info("FactCC model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize FactCC: {e}")
            raise

    def supports_task(self, task_name: str) -> bool:
        """FactCC supports entailment inference and consistency rating."""
        return task_name in ["entailment_inference", "consistency_rating"]

    def _predict_factuality(self, source: str, summary: str) -> Tuple[int, float, Dict]:
        """
        Core FactCC prediction logic using BERT-based model.

        Returns:
            Tuple of (prediction, confidence, raw_scores)
        """
        # For FactCC, we concatenate source and summary with [SEP]
        text = f"{source.strip()} [SEP] {summary.strip()}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # For FactCC: 0=CORRECT, 1=INCORRECT (as per model.config)
        prob_correct = probabilities[0][0].item()
        prob_incorrect = probabilities[0][1].item()

        # Binary prediction: correct = 1, incorrect = 0
        prediction = 1 if prob_correct > prob_incorrect else 0
        confidence = max(prob_correct, prob_incorrect)

        raw_scores = {
            "logits": logits[0].cpu().tolist(),
            "prob_correct": prob_correct,
            "prob_incorrect": prob_incorrect,
            "prediction_logic": "0=CORRECT, 1=INCORRECT",
        }

        return prediction, confidence, raw_scores

    def evaluate_entailment_inference(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """Evaluate entailment inference using FactCC."""
        import time

        # Initialize if not already done
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        prediction, confidence, raw_scores = self._predict_factuality(source, summary)
        processing_time = time.time() - start_time

        return BaselineResult(
            baseline_name="factcc",
            task_name="entailment_inference",
            example_id=example_id or "unknown",
            prediction=prediction,
            confidence=confidence,
            raw_scores=raw_scores,
            processing_time=processing_time,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "prediction_type": "binary_classification",
            },
        )

    def evaluate_summary_ranking(
        self, source: str, summaries: List[str], example_id: str = None
    ) -> BaselineResult:
        """FactCC does not directly support ranking tasks."""
        raise NotImplementedError("FactCC does not support summary ranking tasks")

    def evaluate_consistency_rating(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """Evaluate consistency rating using FactCC probability."""
        import time

        # Initialize if not already done
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        prediction, confidence, raw_scores = self._predict_factuality(source, summary)

        # Convert probability to 0-100 scale
        consistency_rating = raw_scores["prob_correct"] * 100.0
        processing_time = time.time() - start_time

        return BaselineResult(
            baseline_name="factcc",
            task_name="consistency_rating",
            example_id=example_id or "unknown",
            prediction=consistency_rating,
            confidence=confidence,
            raw_scores={**raw_scores, "rating_scale": "0-100"},
            processing_time=processing_time,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "prediction_type": "consistency_rating",
            },
        )


class BERTScoreBaseline(SOTABaseline):
    """
    BERTScore baseline implementation for semantic similarity evaluation.

    BERTScore uses BERT embeddings to compute similarity between reference
    and candidate texts, adapted for all three ChatGPT factuality tasks.
    """

    def __init__(self, config: Optional[Dict] = None, device: Optional[str] = None):
        super().__init__(config, device)

        # BERTScore-specific configuration
        bertscore_config = self.config.get("baselines", {}).get("bertscore", {})
        self.model_type = bertscore_config.get("model_type", "roberta-large")
        self.num_layers = bertscore_config.get("num_layers", 17)
        self.batch_size = bertscore_config.get("batch_size", 16)

    def initialize(self) -> None:
        """BERTScore initialization is handled per-call."""
        self._is_initialized = True
        self.logger.info(f"BERTScore initialized with model: {self.model_type}")

    def supports_task(self, task_name: str) -> bool:
        """BERTScore supports all three tasks."""
        return task_name in [
            "entailment_inference",
            "summary_ranking",
            "consistency_rating",
        ]

    def _compute_bertscore(self, references: List[str], candidates: List[str]) -> Dict:
        """
        Compute BERTScore for given reference-candidate pairs.

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        try:
            # Compute BERTScore
            P, R, F1 = bert_score.score(
                candidates,
                references,
                model_type=self.model_type,
                num_layers=self.num_layers,
                verbose=False,
                device=self.device,
                batch_size=self.batch_size,
            )

            return {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1_score": F1.mean().item(),
                "precision_all": P.tolist(),
                "recall_all": R.tolist(),
                "f1_all": F1.tolist(),
            }

        except Exception as e:
            self.logger.error(f"BERTScore computation failed: {e}")
            # Return default scores on failure
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "error": str(e)}

    def evaluate_entailment_inference(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """Evaluate entailment using BERTScore with adaptive threshold."""
        import time

        # Initialize if not already done
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        scores = self._compute_bertscore([source], [summary])

        # Use F1 score for binary classification
        f1_score = scores.get("f1_score", 0.0)
        
        # Store raw score for later threshold calculation
        prediction = f1_score  # Return raw score, threshold applied later
        confidence = f1_score  # Use raw score as confidence

        processing_time = time.time() - start_time

        return BaselineResult(
            baseline_name="bertscore",
            task_name="entailment_inference",
            example_id=example_id or "unknown",
            prediction=prediction,
            confidence=confidence,
            raw_scores={**scores, "decision_score": f1_score},
            processing_time=processing_time,
            metadata={
                "model_type": self.model_type,
                "num_layers": self.num_layers,
                "prediction_type": "similarity_score",
            },
        )

    def evaluate_summary_ranking(
        self, source: str, summaries: List[str], example_id: str = None
    ) -> BaselineResult:
        """Rank summaries by BERTScore with source."""
        import time

        start_time = time.time()

        # Compute BERTScore for each summary against source
        f1_scores = []
        all_scores = []

        for summary in summaries:
            scores = self._compute_bertscore([source], [summary])
            f1_scores.append(scores.get("f1_score", 0.0))
            all_scores.append(scores)

        # Rank by F1 score (higher is better)
        ranked_indices = sorted(
            range(len(f1_scores)), key=lambda i: f1_scores[i], reverse=True
        )

        # Convert to 1-based ranking as expected by ChatGPT comparison
        ranking = [0] * len(summaries)
        for rank, idx in enumerate(ranked_indices):
            ranking[idx] = rank + 1

        processing_time = time.time() - start_time

        return BaselineResult(
            baseline_name="bertscore",
            task_name="summary_ranking",
            example_id=example_id or "unknown",
            prediction=ranking,
            confidence=np.std(f1_scores),  # Ranking confidence based on score variance
            raw_scores={
                "f1_scores": f1_scores,
                "all_scores": all_scores,
                "ranked_indices": ranked_indices,
            },
            processing_time=processing_time,
            metadata={
                "model_type": self.model_type,
                "num_summaries": len(summaries),
                "prediction_type": "similarity_ranking",
            },
        )

    def evaluate_consistency_rating(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """Rate consistency using BERTScore F1."""
        import time

        # Initialize if not already done
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        scores = self._compute_bertscore([source], [summary])

        # Convert F1 score to 0-100 scale
        f1_score = scores.get("f1_score", 0.0)
        consistency_rating = f1_score * 100.0
        confidence = f1_score  # F1 score itself as confidence

        processing_time = time.time() - start_time

        return BaselineResult(
            baseline_name="bertscore",
            task_name="consistency_rating",
            example_id=example_id or "unknown",
            prediction=consistency_rating,
            confidence=confidence,
            raw_scores={**scores, "rating_scale": "0-100"},
            processing_time=processing_time,
            metadata={
                "model_type": self.model_type,
                "prediction_type": "similarity_rating",
            },
        )


class ROUGEBaseline(SOTABaseline):
    """
    ROUGE baseline implementation for n-gram overlap evaluation.

    ROUGE measures n-gram overlap between reference and candidate texts,
    adapted for summary ranking and consistency rating tasks.
    """

    def __init__(self, config: Optional[Dict] = None, device: Optional[str] = None):
        super().__init__(config, device)

        # ROUGE-specific configuration
        rouge_config = self.config.get("baselines", {}).get("rouge", {})
        self.rouge_types = rouge_config.get(
            "rouge_types", ["rouge1", "rouge2", "rougeL"]
        )
        self.use_stemmer = rouge_config.get("use_stemmer", True)

        # Initialize ROUGE scorer
        self.scorer = rouge_scorer.RougeScorer(
            self.rouge_types, use_stemmer=self.use_stemmer
        )

    def initialize(self) -> None:
        """ROUGE requires no model loading."""
        self._is_initialized = True
        self.logger.info(f"ROUGE initialized with types: {self.rouge_types}")

    def supports_task(self, task_name: str) -> bool:
        """ROUGE supports ranking and rating tasks."""
        return task_name in ["summary_ranking", "consistency_rating"]

    def _compute_rouge_scores(self, reference: str, candidate: str) -> Dict:
        """
        Compute ROUGE scores for reference-candidate pair.

        Returns:
            Dictionary with ROUGE scores
        """
        try:
            scores = self.scorer.score(reference, candidate)

            # Extract F1 scores (primary metric)
            rouge_scores = {}
            for rouge_type in self.rouge_types:
                if rouge_type in scores:
                    rouge_scores[f"{rouge_type}_precision"] = scores[
                        rouge_type
                    ].precision
                    rouge_scores[f"{rouge_type}_recall"] = scores[rouge_type].recall
                    rouge_scores[f"{rouge_type}_fmeasure"] = scores[rouge_type].fmeasure

            # Compute average F1 across ROUGE types
            f1_scores = [scores[rt].fmeasure for rt in self.rouge_types if rt in scores]
            rouge_scores["average_f1"] = np.mean(f1_scores) if f1_scores else 0.0

            return rouge_scores

        except Exception as e:
            self.logger.error(f"ROUGE computation failed: {e}")
            return {"average_f1": 0.0, "error": str(e)}

    def evaluate_entailment_inference(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """ROUGE does not directly support entailment inference."""
        raise NotImplementedError("ROUGE does not support entailment inference tasks")

    def evaluate_summary_ranking(
        self, source: str, summaries: List[str], example_id: str = None
    ) -> BaselineResult:
        """Rank summaries by ROUGE scores with source."""
        import time

        start_time = time.time()

        # Compute ROUGE scores for each summary
        rouge_scores = []
        all_scores = []

        for summary in summaries:
            scores = self._compute_rouge_scores(source, summary)
            rouge_scores.append(scores.get("average_f1", 0.0))
            all_scores.append(scores)

        # Rank by average F1 score (higher is better)
        ranked_indices = sorted(
            range(len(rouge_scores)), key=lambda i: rouge_scores[i], reverse=True
        )

        # Convert to 1-based ranking
        ranking = [0] * len(summaries)
        for rank, idx in enumerate(ranked_indices):
            ranking[idx] = rank + 1

        processing_time = time.time() - start_time

        return BaselineResult(
            baseline_name="rouge",
            task_name="summary_ranking",
            example_id=example_id or "unknown",
            prediction=ranking,
            confidence=np.std(
                rouge_scores
            ),  # Ranking confidence based on score variance
            raw_scores={
                "rouge_scores": rouge_scores,
                "all_scores": all_scores,
                "ranked_indices": ranked_indices,
            },
            processing_time=processing_time,
            metadata={
                "rouge_types": self.rouge_types,
                "use_stemmer": self.use_stemmer,
                "num_summaries": len(summaries),
                "prediction_type": "ngram_ranking",
            },
        )

    def evaluate_consistency_rating(
        self, source: str, summary: str, example_id: str = None
    ) -> BaselineResult:
        """Rate consistency using ROUGE scores."""
        import time

        # Initialize if not already done
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        scores = self._compute_rouge_scores(source, summary)

        # Convert average F1 to 0-100 scale
        average_f1 = scores.get("average_f1", 0.0)
        consistency_rating = average_f1 * 100.0
        confidence = average_f1  # F1 score as confidence

        processing_time = time.time() - start_time

        return BaselineResult(
            baseline_name="rouge",
            task_name="consistency_rating",
            example_id=example_id or "unknown",
            prediction=consistency_rating,
            confidence=confidence,
            raw_scores={**scores, "rating_scale": "0-100"},
            processing_time=processing_time,
            metadata={
                "rouge_types": self.rouge_types,
                "use_stemmer": self.use_stemmer,
                "prediction_type": "ngram_rating",
            },
        )


class BaselineComparator:
    """
    Comprehensive comparator for ChatGPT vs SOTA baseline analysis.

    Provides statistical comparison, correlation analysis, and performance
    evaluation between ChatGPT and traditional factuality metrics.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize baseline comparator."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)

        # Statistical analysis configuration
        stats_config = self.config.get("evaluation", {}).get("statistics", {})
        self.confidence_level = stats_config.get("confidence_level", 0.95)
        self.significance_level = stats_config.get("significance_level", 0.05)

    def compare_predictions(
        self,
        chatgpt_results: List[Dict],
        baseline_results: List[BaselineResult],
        task_name: str,
    ) -> Dict[str, Any]:
        """
        Compare ChatGPT predictions with baseline predictions.

        Args:
            chatgpt_results: List of ChatGPT task results
            baseline_results: List of baseline results
            task_name: Name of the factuality task

        Returns:
            Comprehensive comparison analysis
        """
        self.logger.info(f"Comparing ChatGPT vs baselines for {task_name}")

        # Ensure result alignment
        aligned_results = self._align_results(chatgpt_results, baseline_results)

        if not aligned_results:
            return {"error": "No aligned results found for comparison"}

        # Task-specific comparison
        if task_name == "entailment_inference":
            return self._compare_binary_classification(aligned_results, task_name)
        elif task_name == "summary_ranking":
            return self._compare_rankings(aligned_results, task_name)
        elif task_name == "consistency_rating":
            return self._compare_ratings(aligned_results, task_name)
        else:
            raise ValueError(f"Unknown task for comparison: {task_name}")

    def _align_results(
        self, chatgpt_results: List[Dict], baseline_results: List[BaselineResult]
    ) -> List[Tuple[Dict, BaselineResult]]:
        """Align ChatGPT and baseline results by example ID."""
        # Create lookup for baseline results
        baseline_lookup = {br.example_id: br for br in baseline_results}

        aligned = []
        for cgpt_result in chatgpt_results:
            example_id = cgpt_result.get("example_id")
            if example_id in baseline_lookup:
                aligned.append((cgpt_result, baseline_lookup[example_id]))

        self.logger.info(f"Aligned {len(aligned)} result pairs for comparison")
        return aligned

    def _compare_binary_classification(
        self, aligned_results: List[Tuple[Dict, BaselineResult]], task_name: str
    ) -> Dict[str, Any]:
        """Compare binary classification results."""
        chatgpt_preds = []
        baseline_preds = []

        for cgpt_result, baseline_result in aligned_results:
            # Extract binary predictions
            cgpt_pred = int(cgpt_result.get("prediction", 0))
            baseline_pred = int(baseline_result.prediction)

            chatgpt_preds.append(cgpt_pred)
            baseline_preds.append(baseline_pred)

        # Compute agreement metrics
        agreement = np.mean([cp == bp for cp, bp in zip(chatgpt_preds, baseline_preds)])

        # McNemar's test for paired binary comparisons
        contingency_table = self._build_contingency_table(chatgpt_preds, baseline_preds)
        mcnemar_p_value = self._mcnemar_test(contingency_table)

        # Cohen's Kappa for inter-rater agreement
        kappa = self._cohens_kappa(chatgpt_preds, baseline_preds)

        return {
            "task_name": task_name,
            "comparison_type": "binary_classification",
            "num_examples": len(aligned_results),
            "agreement_rate": float(agreement),
            "cohens_kappa": float(kappa),
            "mcnemar_p_value": float(mcnemar_p_value),
            "contingency_table": contingency_table,
            "chatgpt_predictions": chatgpt_preds,
            "baseline_predictions": baseline_preds,
            "statistical_significance": mcnemar_p_value < self.significance_level,
        }

    def _compare_rankings(
        self, aligned_results: List[Tuple[Dict, BaselineResult]], task_name: str
    ) -> Dict[str, Any]:
        """Compare ranking results."""
        kendall_correlations = []
        spearman_correlations = []

        for cgpt_result, baseline_result in aligned_results:
            # Extract rankings (assuming both are lists of rankings)
            cgpt_ranking = cgpt_result.get("prediction", [])
            baseline_ranking = baseline_result.prediction

            if len(cgpt_ranking) == len(baseline_ranking) and len(cgpt_ranking) > 1:
                # Compute rank correlations
                kendall_tau, kendall_p = stats.kendalltau(
                    cgpt_ranking, baseline_ranking
                )
                spearman_rho, spearman_p = stats.spearmanr(
                    cgpt_ranking, baseline_ranking
                )

                kendall_correlations.append(kendall_tau)
                spearman_correlations.append(spearman_rho)

        # Average correlations
        avg_kendall = np.mean(kendall_correlations) if kendall_correlations else 0.0
        avg_spearman = np.mean(spearman_correlations) if spearman_correlations else 0.0

        return {
            "task_name": task_name,
            "comparison_type": "ranking_correlation",
            "num_examples": len(aligned_results),
            "avg_kendall_tau": float(avg_kendall),
            "avg_spearman_rho": float(avg_spearman),
            "kendall_correlations": kendall_correlations,
            "spearman_correlations": spearman_correlations,
            "correlation_strength": self._interpret_correlation(avg_spearman),
        }

    def _compare_ratings(
        self, aligned_results: List[Tuple[Dict, BaselineResult]], task_name: str
    ) -> Dict[str, Any]:
        """Compare rating results."""
        chatgpt_ratings = []
        baseline_ratings = []

        for cgpt_result, baseline_result in aligned_results:
            # Extract numerical ratings
            cgpt_rating = float(cgpt_result.get("prediction", 0))
            baseline_rating = float(baseline_result.prediction)

            chatgpt_ratings.append(cgpt_rating)
            baseline_ratings.append(baseline_rating)

        # Correlation analysis
        pearson_r, pearson_p = stats.pearsonr(chatgpt_ratings, baseline_ratings)
        spearman_rho, spearman_p = stats.spearmanr(chatgpt_ratings, baseline_ratings)

        # Mean Absolute Error
        mae = np.mean(np.abs(np.array(chatgpt_ratings) - np.array(baseline_ratings)))

        # Root Mean Square Error
        rmse = np.sqrt(
            np.mean((np.array(chatgpt_ratings) - np.array(baseline_ratings)) ** 2)
        )

        return {
            "task_name": task_name,
            "comparison_type": "rating_correlation",
            "num_examples": len(aligned_results),
            "pearson_correlation": float(pearson_r),
            "pearson_p_value": float(pearson_p),
            "spearman_correlation": float(spearman_rho),
            "spearman_p_value": float(spearman_p),
            "mean_absolute_error": float(mae),
            "root_mean_square_error": float(rmse),
            "chatgpt_ratings": chatgpt_ratings,
            "baseline_ratings": baseline_ratings,
            "statistical_significance": pearson_p < self.significance_level,
        }

    def _build_contingency_table(
        self, pred1: List[int], pred2: List[int]
    ) -> List[List[int]]:
        """Build 2x2 contingency table for McNemar's test."""
        # pred1 = ChatGPT, pred2 = Baseline
        # [[both_0, cgpt_0_baseline_1], [cgpt_1_baseline_0, both_1]]
        both_0 = sum(1 for p1, p2 in zip(pred1, pred2) if p1 == 0 and p2 == 0)
        cgpt_0_baseline_1 = sum(
            1 for p1, p2 in zip(pred1, pred2) if p1 == 0 and p2 == 1
        )
        cgpt_1_baseline_0 = sum(
            1 for p1, p2 in zip(pred1, pred2) if p1 == 1 and p2 == 0
        )
        both_1 = sum(1 for p1, p2 in zip(pred1, pred2) if p1 == 1 and p2 == 1)

        return [[both_0, cgpt_0_baseline_1], [cgpt_1_baseline_0, both_1]]

    def _mcnemar_test(self, contingency_table: List[List[int]]) -> float:
        """Perform McNemar's test for paired binary data."""
        # McNemar's test: focuses on discordant pairs
        b = contingency_table[0][1]  # ChatGPT=0, Baseline=1
        c = contingency_table[1][0]  # ChatGPT=1, Baseline=0

        if b + c == 0:
            return 1.0  # No discordant pairs, perfect agreement

        # McNemar's test statistic with continuity correction
        chi_square = (abs(b - c) - 1) ** 2 / (b + c)

        # Convert to p-value (1 degree of freedom)
        from scipy.stats import chi2

        p_value = 1 - chi2.cdf(chi_square, 1)

        return p_value

    def _cohens_kappa(self, pred1: List[int], pred2: List[int]) -> float:
        """Compute Cohen's Kappa for inter-rater agreement."""
        # Observed agreement
        po = np.mean([p1 == p2 for p1, p2 in zip(pred1, pred2)])

        # Expected agreement by chance
        p1_pos = np.mean(pred1)
        p2_pos = np.mean(pred2)
        pe = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

        # Cohen's Kappa
        if pe == 1.0:
            return 1.0  # Perfect agreement scenario

        kappa = (po - pe) / (1 - pe)
        return kappa

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"


# Factory functions and utilities


def create_baseline(baseline_name: str, config: Optional[Dict] = None) -> SOTABaseline:
    """
    Factory function to create baseline instances.

    Args:
        baseline_name: Name of the baseline to create
        config: Configuration dictionary

    Returns:
        Initialized baseline instance

    Raises:
        ValueError: If baseline name is not recognized
    """
    baseline_classes = {
        "factcc": FactCCBaseline,
        "bertscore": BERTScoreBaseline,
        "rouge": ROUGEBaseline,
    }

    if baseline_name not in baseline_classes:
        available = list(baseline_classes.keys())
        raise ValueError(f"Unknown baseline '{baseline_name}'. Available: {available}")

    return baseline_classes[baseline_name](config)


def get_available_baselines() -> List[str]:
    """Get list of available baseline method names."""
    return ["factcc", "bertscore", "rouge"]


def adapt_data_for_baseline(
    examples: List[FactualityExample], task_name: str
) -> List[FactualityExample]:
    """
    Adapt data examples for baseline evaluation.

    Args:
        examples: List of data examples
        task_name: Name of the factuality task

    Returns:
        Adapted examples suitable for baseline evaluation
    """
    # For now, return examples as-is
    # Could add task-specific data preprocessing in the future
    return examples


def compare_with_chatgpt(
    chatgpt_results: List[Dict],
    baseline_results: Dict[str, List[BaselineResult]],
    task_name: str,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Compare ChatGPT results with all available baselines.

    Args:
        chatgpt_results: List of ChatGPT task results
        baseline_results: Dictionary mapping baseline names to their results
        task_name: Name of the factuality task
        config: Configuration dictionary

    Returns:
        Comprehensive comparison across all baselines
    """
    comparator = BaselineComparator(config)

    all_comparisons = {}
    for baseline_name, results in baseline_results.items():
        comparison = comparator.compare_predictions(chatgpt_results, results, task_name)
        all_comparisons[baseline_name] = comparison

    # Add summary statistics
    all_comparisons["summary"] = {
        "task_name": task_name,
        "num_baselines": len(baseline_results),
        "baselines_compared": list(baseline_results.keys()),
        "total_examples": len(chatgpt_results),
    }

    return all_comparisons


def generate_baseline_report(
    comparison_results: Dict[str, Any], output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive baseline comparison report.

    Args:
        comparison_results: Results from compare_with_chatgpt
        output_path: Optional path to save report

    Returns:
        Formatted report with key findings
    """
    report = {
        "report_metadata": {
            "generated_at": str(time.time()),
            "task_name": comparison_results.get("summary", {}).get(
                "task_name", "unknown"
            ),
            "num_baselines": comparison_results.get("summary", {}).get(
                "num_baselines", 0
            ),
        },
        "key_findings": [],
        "detailed_results": comparison_results,
    }

    # Extract key findings for each baseline
    for baseline_name, results in comparison_results.items():
        if baseline_name == "summary":
            continue

        finding = {
            "baseline": baseline_name,
            "comparison_type": results.get("comparison_type", "unknown"),
        }

        if results.get("comparison_type") == "binary_classification":
            finding.update(
                {
                    "agreement_rate": results.get("agreement_rate", 0.0),
                    "cohens_kappa": results.get("cohens_kappa", 0.0),
                    "statistically_significant": results.get(
                        "statistical_significance", False
                    ),
                }
            )
        elif results.get("comparison_type") == "rating_correlation":
            finding.update(
                {
                    "pearson_correlation": results.get("pearson_correlation", 0.0),
                    "mae": results.get("mean_absolute_error", 0.0),
                    "statistically_significant": results.get(
                        "statistical_significance", False
                    ),
                }
            )
        elif results.get("comparison_type") == "ranking_correlation":
            finding.update(
                {
                    "kendall_tau": results.get("avg_kendall_tau", 0.0),
                    "spearman_rho": results.get("avg_spearman_rho", 0.0),
                    "correlation_strength": results.get(
                        "correlation_strength", "unknown"
                    ),
                }
            )

        report["key_findings"].append(finding)

    # Save report if path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Baseline comparison report saved to: {output_path}")

    return report
