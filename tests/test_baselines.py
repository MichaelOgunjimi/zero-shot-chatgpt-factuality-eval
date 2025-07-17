"""
Baseline Methods and SOTA Metrics Tests
========================================

Tests for baseline factuality evaluation methods and state-of-the-art
metrics including ROUGE, BERTScore, and other automated evaluation methods.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import json
from pathlib import Path


class FactualityExample:
    """Data model for factuality examples"""
    
    def __init__(self, article=None, summary=None, dataset_name=None, 
                 split=None, index=None, metadata=None):
        self.article = article or ""
        self.summary = summary or ""
        self.dataset_name = dataset_name or ""
        self.split = split or "test"
        self.index = index or 0
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            "article": self.article,
            "summary": self.summary,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "index": self.index,
            "metadata": self.metadata
        }


class ROUGEMetric:
    """ROUGE evaluation metric implementation"""
    
    def __init__(self, rouge_types=None, use_stemmer=True):
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.use_stemmer = use_stemmer
    
    def compute_rouge(self, reference, hypothesis):
        """Compute ROUGE scores between reference and hypothesis"""
        if not reference or not hypothesis:
            return {rouge_type: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} 
                   for rouge_type in self.rouge_types}
        
        # Tokenize texts
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(hypothesis)
        
        scores = {}
        
        for rouge_type in self.rouge_types:
            if rouge_type == 'rouge1':
                scores[rouge_type] = self._compute_rouge_n(ref_tokens, hyp_tokens, 1)
            elif rouge_type == 'rouge2':
                scores[rouge_type] = self._compute_rouge_n(ref_tokens, hyp_tokens, 2)
            elif rouge_type == 'rougeL':
                scores[rouge_type] = self._compute_rouge_l(ref_tokens, hyp_tokens)
        
        return scores
    
    def _tokenize(self, text):
        """Simple tokenization"""
        import re
        text = text.lower()
        if self.use_stemmer:
            text = self._simple_stem(text)
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _simple_stem(self, text):
        """Simple stemming (removes common suffixes)"""
        import re
        # Simple suffix removal
        text = re.sub(r'ing\b', '', text)
        text = re.sub(r'ed\b', '', text)
        text = re.sub(r'er\b', '', text)
        text = re.sub(r'est\b', '', text)
        return text
    
    def _compute_rouge_n(self, ref_tokens, hyp_tokens, n):
        """Compute ROUGE-N scores"""
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        hyp_ngrams = self._get_ngrams(hyp_tokens, n)
        
        if not ref_ngrams and not hyp_ngrams:
            return {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}
        elif not ref_ngrams or not hyp_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
        
        overlap = len(ref_ngrams & hyp_ngrams)
        
        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        
        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)
        
        return {'precision': precision, 'recall': recall, 'fmeasure': fmeasure}
    
    def _compute_rouge_l(self, ref_tokens, hyp_tokens):
        """Compute ROUGE-L scores using LCS"""
        lcs_length = self._lcs_length(ref_tokens, hyp_tokens)
        
        if len(ref_tokens) == 0 and len(hyp_tokens) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}
        elif len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
        
        precision = lcs_length / len(hyp_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)
        
        return {'precision': precision, 'recall': recall, 'fmeasure': fmeasure}
    
    def _get_ngrams(self, tokens, n):
        """Get n-grams from tokens"""
        if len(tokens) < n:
            return set()
        
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i:i+n]))
        
        return ngrams
    
    def _lcs_length(self, seq1, seq2):
        """Compute length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0
        
        # Dynamic programming approach
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def compute_batch(self, references, hypotheses):
        """Compute ROUGE scores for batch of examples"""
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        batch_scores = []
        for ref, hyp in zip(references, hypotheses):
            scores = self.compute_rouge(ref, hyp)
            batch_scores.append(scores)
        
        return batch_scores
    
    def aggregate_scores(self, batch_scores):
        """Aggregate scores across a batch"""
        if not batch_scores:
            return {}
        
        aggregated = {}
        
        for rouge_type in self.rouge_types:
            metrics = ['precision', 'recall', 'fmeasure']
            aggregated[rouge_type] = {}
            
            for metric in metrics:
                values = [scores[rouge_type][metric] for scores in batch_scores]
                aggregated[rouge_type][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated


class BERTScoreMetric:
    """BERTScore evaluation metric (simplified implementation)"""
    
    def __init__(self, model_type="bert-base-uncased", batch_size=32):
        self.model_type = model_type
        self.batch_size = batch_size
        self._embeddings_cache = {}
    
    def compute_bertscore(self, reference, hypothesis):
        """Compute BERTScore between reference and hypothesis"""
        if not reference or not hypothesis:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Get embeddings (mock implementation)
        ref_embeddings = self._get_embeddings(reference)
        hyp_embeddings = self._get_embeddings(hypothesis)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(ref_embeddings, hyp_embeddings)
        
        # Compute precision, recall, F1
        precision = np.max(similarity_matrix, axis=0).mean()
        recall = np.max(similarity_matrix, axis=1).mean()
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
    
    def _get_embeddings(self, text):
        """Get BERT embeddings for text (mock implementation)"""
        # Cache embeddings
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]
        
        # Mock embeddings - in reality would use BERT model
        words = text.lower().split()
        embeddings = []
        
        for word in words:
            # Create pseudo-embedding based on word hash
            word_hash = hash(word) % 1000
            embedding = np.random.RandomState(word_hash).randn(768)  # BERT-base dimension
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        self._embeddings_cache[text] = embeddings
        
        return embeddings
    
    def _compute_similarity_matrix(self, ref_embeddings, hyp_embeddings):
        """Compute cosine similarity matrix between embeddings"""
        # Normalize embeddings
        ref_norm = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
        hyp_norm = hyp_embeddings / np.linalg.norm(hyp_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(ref_norm, hyp_norm.T)
        
        return similarity_matrix
    
    def compute_batch(self, references, hypotheses):
        """Compute BERTScore for batch of examples"""
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        batch_scores = []
        for ref, hyp in zip(references, hypotheses):
            scores = self.compute_bertscore(ref, hyp)
            batch_scores.append(scores)
        
        return batch_scores


class FactCheckBaseline:
    """Baseline fact-checking method using keyword matching"""
    
    def __init__(self, factual_keywords=None):
        self.factual_keywords = factual_keywords or [
            'said', 'reported', 'according to', 'data shows', 'study found',
            'research indicates', 'statistics show', 'evidence suggests'
        ]
    
    def evaluate_factuality(self, article, summary):
        """Evaluate factuality using simple keyword-based approach"""
        if not article or not summary:
            return {'factuality_score': 0.0, 'explanation': 'Empty input'}
        
        article_lower = article.lower()
        summary_lower = summary.lower()
        
        # Count factual indicators in summary
        factual_indicators = 0
        for keyword in self.factual_keywords:
            if keyword in summary_lower:
                factual_indicators += 1
        
        # Simple scoring based on factual indicators and length
        base_score = min(factual_indicators / len(self.factual_keywords), 1.0)
        
        # Penalize if summary is much longer than reasonable
        length_penalty = 0.0
        if len(summary.split()) > len(article.split()) * 0.3:
            length_penalty = 0.2
        
        factuality_score = max(0.0, base_score - length_penalty)
        
        explanation = f"Found {factual_indicators} factual indicators"
        if length_penalty > 0:
            explanation += f", applied length penalty of {length_penalty}"
        
        return {
            'factuality_score': factuality_score,
            'explanation': explanation,
            'factual_indicators': factual_indicators
        }
    
    def evaluate_batch(self, examples):
        """Evaluate factuality for batch of examples"""
        batch_results = []
        
        for example in examples:
            if hasattr(example, 'article') and hasattr(example, 'summary'):
                result = self.evaluate_factuality(example.article, example.summary)
            else:
                result = {'factuality_score': 0.0, 'explanation': 'Invalid example format'}
            
            batch_results.append(result)
        
        return batch_results


class NamedEntityConsistencyBaseline:
    """Baseline method checking named entity consistency"""
    
    def __init__(self):
        self.entity_patterns = {
            'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'organization': r'\b[A-Z][A-Z]+\b|\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC)\b',
            'location': r'\b[A-Z][a-z]+(?:, [A-Z][a-z]+)*\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b[A-Z][a-z]+ \d{1,2}, \d{4}\b',
            'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        }
    
    def extract_entities(self, text):
        """Extract named entities using simple regex patterns"""
        import re
        
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            entities[entity_type] = set(matches)
        
        return entities
    
    def evaluate_consistency(self, article, summary):
        """Evaluate named entity consistency between article and summary"""
        if not article or not summary:
            return {'consistency_score': 0.0, 'explanation': 'Empty input'}
        
        article_entities = self.extract_entities(article)
        summary_entities = self.extract_entities(summary)
        
        consistency_scores = {}
        total_score = 0.0
        total_weight = 0.0
        
        entity_weights = {
            'person': 0.3,
            'organization': 0.25,
            'location': 0.2,
            'date': 0.15,
            'number': 0.1
        }
        
        for entity_type in self.entity_patterns.keys():
            article_set = article_entities[entity_type]
            summary_set = summary_entities[entity_type]
            
            if not summary_set:
                # No entities in summary for this type
                consistency_scores[entity_type] = 1.0 if not article_set else 0.8
            else:
                # Calculate overlap
                overlap = len(article_set & summary_set)
                consistency = overlap / len(summary_set)
                consistency_scores[entity_type] = consistency
            
            weight = entity_weights[entity_type]
            total_score += consistency_scores[entity_type] * weight
            total_weight += weight
        
        overall_consistency = total_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'consistency_score': overall_consistency,
            'entity_scores': consistency_scores,
            'article_entities': {k: list(v) for k, v in article_entities.items()},
            'summary_entities': {k: list(v) for k, v in summary_entities.items()}
        }
    
    def evaluate_batch(self, examples):
        """Evaluate entity consistency for batch of examples"""
        batch_results = []
        
        for example in examples:
            if hasattr(example, 'article') and hasattr(example, 'summary'):
                result = self.evaluate_consistency(example.article, example.summary)
            else:
                result = {'consistency_score': 0.0, 'explanation': 'Invalid example format'}
            
            batch_results.append(result)
        
        return batch_results


class SOTAMetricsEvaluator:
    """Evaluator combining multiple SOTA metrics"""
    
    def __init__(self, use_rouge=True, use_bertscore=True, use_factcheck=True, use_entity_consistency=True):
        self.metrics = {}
        
        if use_rouge:
            self.metrics['rouge'] = ROUGEMetric()
        
        if use_bertscore:
            self.metrics['bertscore'] = BERTScoreMetric()
        
        if use_factcheck:
            self.metrics['factcheck'] = FactCheckBaseline()
        
        if use_entity_consistency:
            self.metrics['entity_consistency'] = NamedEntityConsistencyBaseline()
    
    def evaluate_example(self, example):
        """Evaluate single example with all metrics"""
        if not hasattr(example, 'article') or not hasattr(example, 'summary'):
            return {'error': 'Invalid example format'}
        
        results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                if metric_name == 'rouge':
                    scores = metric.compute_rouge(example.article, example.summary)
                    results[metric_name] = scores
                
                elif metric_name == 'bertscore':
                    scores = metric.compute_bertscore(example.article, example.summary)
                    results[metric_name] = scores
                
                elif metric_name == 'factcheck':
                    scores = metric.evaluate_factuality(example.article, example.summary)
                    results[metric_name] = scores
                
                elif metric_name == 'entity_consistency':
                    scores = metric.evaluate_consistency(example.article, example.summary)
                    results[metric_name] = scores
                
            except Exception as e:
                results[metric_name] = {'error': str(e)}
        
        return results
    
    def evaluate_batch(self, examples):
        """Evaluate batch of examples with all metrics"""
        batch_results = []
        
        for example in examples:
            result = self.evaluate_example(example)
            batch_results.append(result)
        
        return batch_results
    
    def aggregate_results(self, batch_results):
        """Aggregate results across batch"""
        if not batch_results:
            return {}
        
        aggregated = {}
        
        for metric_name in self.metrics.keys():
            metric_results = []
            
            for result in batch_results:
                if metric_name in result and 'error' not in result[metric_name]:
                    metric_results.append(result[metric_name])
            
            if metric_results:
                aggregated[metric_name] = self._aggregate_metric_results(metric_name, metric_results)
        
        return aggregated
    
    def _aggregate_metric_results(self, metric_name, results):
        """Aggregate results for specific metric"""
        if metric_name == 'rouge':
            return self._aggregate_rouge_results(results)
        
        elif metric_name == 'bertscore':
            return self._aggregate_bertscore_results(results)
        
        elif metric_name == 'factcheck':
            scores = [r['factuality_score'] for r in results]
            return {
                'mean_factuality_score': np.mean(scores),
                'std_factuality_score': np.std(scores)
            }
        
        elif metric_name == 'entity_consistency':
            scores = [r['consistency_score'] for r in results]
            return {
                'mean_consistency_score': np.mean(scores),
                'std_consistency_score': np.std(scores)
            }
        
        return {}
    
    def _aggregate_rouge_results(self, results):
        """Aggregate ROUGE results"""
        aggregated = {}
        
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            if rouge_type in results[0]:
                metrics = ['precision', 'recall', 'fmeasure']
                aggregated[rouge_type] = {}
                
                for metric in metrics:
                    values = [r[rouge_type][metric] for r in results]
                    aggregated[rouge_type][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        
        return aggregated
    
    def _aggregate_bertscore_results(self, results):
        """Aggregate BERTScore results"""
        metrics = ['precision', 'recall', 'f1']
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return aggregated


class BaselineComparator:
    """Compare different baseline methods"""
    
    def __init__(self):
        self.baselines = {
            'rouge': ROUGEMetric(),
            'bertscore': BERTScoreMetric(),
            'factcheck': FactCheckBaseline(),
            'entity_consistency': NamedEntityConsistencyBaseline()
        }
        self.comparison_results = {}
    
    def compare_baselines(self, examples, ground_truth_scores=None):
        """Compare all baseline methods on examples"""
        results = {}
        
        for baseline_name, baseline in self.baselines.items():
            try:
                if baseline_name == 'rouge':
                    references = [ex.article for ex in examples]
                    hypotheses = [ex.summary for ex in examples]
                    scores = baseline.compute_batch(references, hypotheses)
                    # Extract F-measure for comparison
                    results[baseline_name] = [s['rouge1']['fmeasure'] for s in scores]
                
                elif baseline_name == 'bertscore':
                    references = [ex.article for ex in examples]
                    hypotheses = [ex.summary for ex in examples]
                    scores = baseline.compute_batch(references, hypotheses)
                    results[baseline_name] = [s['f1'] for s in scores]
                
                elif baseline_name == 'factcheck':
                    scores = baseline.evaluate_batch(examples)
                    results[baseline_name] = [s['factuality_score'] for s in scores]
                
                elif baseline_name == 'entity_consistency':
                    scores = baseline.evaluate_batch(examples)
                    results[baseline_name] = [s['consistency_score'] for s in scores]
                
            except Exception as e:
                results[baseline_name] = [0.0] * len(examples)
        
        # Calculate correlations if ground truth provided
        if ground_truth_scores:
            correlations = self._calculate_correlations(results, ground_truth_scores)
            results['correlations'] = correlations
        
        self.comparison_results = results
        return results
    
    def _calculate_correlations(self, baseline_results, ground_truth):
        """Calculate correlations with ground truth scores"""
        correlations = {}
        
        for baseline_name, scores in baseline_results.items():
            if len(scores) == len(ground_truth):
                correlation = np.corrcoef(scores, ground_truth)[0, 1]
                correlations[baseline_name] = correlation if not np.isnan(correlation) else 0.0
        
        return correlations
    
    def get_best_baseline(self, metric='correlation'):
        """Get best performing baseline"""
        if not self.comparison_results:
            return None
        
        if metric == 'correlation' and 'correlations' in self.comparison_results:
            correlations = self.comparison_results['correlations']
            best_baseline = max(correlations.items(), key=lambda x: abs(x[1]))
            return best_baseline[0], best_baseline[1]
        
        return None
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        if not self.comparison_results:
            return "No comparison results available"
        
        report = "Baseline Comparison Report\n"
        report += "=" * 30 + "\n\n"
        
        for baseline_name, scores in self.comparison_results.items():
            if baseline_name == 'correlations':
                continue
            
            report += f"{baseline_name.upper()}:\n"
            report += f"  Mean Score: {np.mean(scores):.4f}\n"
            report += f"  Std Score: {np.std(scores):.4f}\n"
            report += f"  Min Score: {np.min(scores):.4f}\n"
            report += f"  Max Score: {np.max(scores):.4f}\n\n"
        
        if 'correlations' in self.comparison_results:
            report += "CORRELATIONS WITH GROUND TRUTH:\n"
            for baseline, corr in self.comparison_results['correlations'].items():
                report += f"  {baseline}: {corr:.4f}\n"
        
        return report


class TestROUGEMetric:
    """Test ROUGE metric implementation"""
    
    def test_rouge_initialization(self):
        """Test ROUGE metric initialization"""
        rouge = ROUGEMetric()
        
        assert 'rouge1' in rouge.rouge_types
        assert 'rouge2' in rouge.rouge_types
        assert 'rougeL' in rouge.rouge_types
        assert rouge.use_stemmer is True
    
    def test_rouge_custom_types(self):
        """Test ROUGE with custom types"""
        rouge = ROUGEMetric(rouge_types=['rouge1'], use_stemmer=False)
        
        assert rouge.rouge_types == ['rouge1']
        assert rouge.use_stemmer is False
    
    def test_compute_rouge_identical(self):
        """Test ROUGE computation for identical texts"""
        rouge = ROUGEMetric()
        
        text = "This is a test sentence."
        scores = rouge.compute_rouge(text, text)
        
        assert scores['rouge1']['precision'] == 1.0
        assert scores['rouge1']['recall'] == 1.0
        assert scores['rouge1']['fmeasure'] == 1.0
    
    def test_compute_rouge_partial_overlap(self):
        """Test ROUGE computation for partial overlap"""
        rouge = ROUGEMetric()
        
        reference = "The cat sat on the mat."
        hypothesis = "The cat was on the mat."
        
        scores = rouge.compute_rouge(reference, hypothesis)
        
        # Should have some overlap but not perfect
        assert 0.0 < scores['rouge1']['precision'] < 1.0
        assert 0.0 < scores['rouge1']['recall'] < 1.0
        assert 0.0 < scores['rouge1']['fmeasure'] < 1.0
    
    def test_compute_rouge_no_overlap(self):
        """Test ROUGE computation for no overlap"""
        rouge = ROUGEMetric()
        
        reference = "The cat sat on the mat."
        hypothesis = "Dogs are running quickly."
        
        scores = rouge.compute_rouge(reference, hypothesis)
        
        assert scores['rouge1']['precision'] == 0.0
        assert scores['rouge1']['recall'] == 0.0
        assert scores['rouge1']['fmeasure'] == 0.0
    
    def test_compute_rouge_empty_input(self):
        """Test ROUGE computation for empty input"""
        rouge = ROUGEMetric()
        
        scores = rouge.compute_rouge("", "test")
        
        for rouge_type in rouge.rouge_types:
            assert scores[rouge_type]['precision'] == 0.0
            assert scores[rouge_type]['recall'] == 0.0
            assert scores[rouge_type]['fmeasure'] == 0.0
    
    def test_compute_rouge_batch(self):
        """Test ROUGE computation for batch"""
        rouge = ROUGEMetric()
        
        references = ["The cat sat.", "Dogs are running."]
        hypotheses = ["The cat was sitting.", "Dogs run quickly."]
        
        batch_scores = rouge.compute_batch(references, hypotheses)
        
        assert len(batch_scores) == 2
        assert all('rouge1' in scores for scores in batch_scores)
    
    def test_aggregate_scores(self):
        """Test score aggregation"""
        rouge = ROUGEMetric()
        
        batch_scores = [
            {'rouge1': {'precision': 0.8, 'recall': 0.7, 'fmeasure': 0.75},
             'rouge2': {'precision': 0.6, 'recall': 0.5, 'fmeasure': 0.55},
             'rougeL': {'precision': 0.7, 'recall': 0.6, 'fmeasure': 0.65}},
            {'rouge1': {'precision': 0.6, 'recall': 0.9, 'fmeasure': 0.72},
             'rouge2': {'precision': 0.4, 'recall': 0.7, 'fmeasure': 0.51},
             'rougeL': {'precision': 0.5, 'recall': 0.8, 'fmeasure': 0.62}}
        ]
        
        aggregated = rouge.aggregate_scores(batch_scores)
        
        assert 'rouge1' in aggregated
        assert aggregated['rouge1']['precision']['mean'] == 0.7
        assert 'std' in aggregated['rouge1']['precision']


class TestBERTScoreMetric:
    """Test BERTScore metric implementation"""
    
    def test_bertscore_initialization(self):
        """Test BERTScore initialization"""
        bertscore = BERTScoreMetric()
        
        assert bertscore.model_type == "bert-base-uncased"
        assert bertscore.batch_size == 32
        assert len(bertscore._embeddings_cache) == 0
    
    def test_compute_bertscore_identical(self):
        """Test BERTScore for identical texts"""
        bertscore = BERTScoreMetric()
        
        text = "This is a test sentence."
        scores = bertscore.compute_bertscore(text, text)
        
        assert scores['precision'] > 0.9  # Should be very high for identical text
        assert scores['recall'] > 0.9
        assert scores['f1'] > 0.9
    
    def test_compute_bertscore_different(self):
        """Test BERTScore for different texts"""
        bertscore = BERTScoreMetric()
        
        reference = "The cat sat on the mat."
        hypothesis = "Dogs are running in the park."
        
        scores = bertscore.compute_bertscore(reference, hypothesis)
        
        assert 0.0 <= scores['precision'] <= 1.0
        assert 0.0 <= scores['recall'] <= 1.0
        assert 0.0 <= scores['f1'] <= 1.0
    
    def test_compute_bertscore_empty(self):
        """Test BERTScore for empty input"""
        bertscore = BERTScoreMetric()
        
        scores = bertscore.compute_bertscore("", "test")
        
        assert scores['precision'] == 0.0
        assert scores['recall'] == 0.0
        assert scores['f1'] == 0.0
    
    def test_embeddings_caching(self):
        """Test that embeddings are cached"""
        bertscore = BERTScoreMetric()
        
        text = "This is a test."
        
        # First call
        bertscore._get_embeddings(text)
        cache_size_1 = len(bertscore._embeddings_cache)
        
        # Second call (should use cache)
        bertscore._get_embeddings(text)
        cache_size_2 = len(bertscore._embeddings_cache)
        
        assert cache_size_1 == cache_size_2 == 1
    
    def test_compute_batch(self):
        """Test BERTScore batch computation"""
        bertscore = BERTScoreMetric()
        
        references = ["The cat sat.", "Dogs are running."]
        hypotheses = ["The cat was sitting.", "Dogs run quickly."]
        
        batch_scores = bertscore.compute_batch(references, hypotheses)
        
        assert len(batch_scores) == 2
        assert all('precision' in scores for scores in batch_scores)
        assert all('recall' in scores for scores in batch_scores)
        assert all('f1' in scores for scores in batch_scores)


class TestFactCheckBaseline:
    """Test fact-checking baseline"""
    
    def test_factcheck_initialization(self):
        """Test fact-check baseline initialization"""
        factcheck = FactCheckBaseline()
        
        assert len(factcheck.factual_keywords) > 0
        assert 'said' in factcheck.factual_keywords
    
    def test_evaluate_factuality_high_score(self):
        """Test factuality evaluation with high score"""
        factcheck = FactCheckBaseline()
        
        article = "According to the study, researchers found significant results."
        summary = "The study found that data shows important findings, according to researchers."
        
        result = factcheck.evaluate_factuality(article, summary)
        
        assert result['factuality_score'] > 0.0
        assert result['factual_indicators'] > 0
        assert 'explanation' in result
    
    def test_evaluate_factuality_low_score(self):
        """Test factuality evaluation with low score"""
        factcheck = FactCheckBaseline()
        
        article = "This is a simple article."
        summary = "This is a simple summary without factual indicators."
        
        result = factcheck.evaluate_factuality(article, summary)
        
        assert result['factuality_score'] >= 0.0
        assert result['factual_indicators'] == 0
    
    def test_evaluate_factuality_length_penalty(self):
        """Test factuality evaluation with length penalty"""
        factcheck = FactCheckBaseline()
        
        article = "Short article."
        summary = "Very long summary that is much longer than it should be relative to the article length which should trigger a penalty."
        
        result = factcheck.evaluate_factuality(article, summary)
        
        # Length penalty should be applied
        assert 'penalty' in result['explanation'].lower()
    
    def test_evaluate_batch(self):
        """Test batch factuality evaluation"""
        factcheck = FactCheckBaseline()
        
        examples = [
            FactualityExample(
                article="Article with content according to sources.",
                summary="Summary said to be accurate."
            ),
            FactualityExample(
                article="Another article.",
                summary="Another summary."
            )
        ]
        
        results = factcheck.evaluate_batch(examples)
        
        assert len(results) == 2
        assert all('factuality_score' in result for result in results)


class TestNamedEntityConsistencyBaseline:
    """Test named entity consistency baseline"""
    
    def test_entity_consistency_initialization(self):
        """Test entity consistency initialization"""
        entity_baseline = NamedEntityConsistencyBaseline()
        
        assert 'person' in entity_baseline.entity_patterns
        assert 'organization' in entity_baseline.entity_patterns
        assert 'date' in entity_baseline.entity_patterns
    
    def test_extract_entities(self):
        """Test entity extraction"""
        entity_baseline = NamedEntityConsistencyBaseline()
        
        text = "John Smith works at Microsoft Corp. He was born on 01/15/1990 and earned $50,000."
        entities = entity_baseline.extract_entities(text)
        
        assert len(entities['person']) > 0
        assert len(entities['number']) > 0
        # Note: Microsoft Corp might not match the simple regex
    
    def test_evaluate_consistency_perfect(self):
        """Test consistency evaluation with perfect consistency"""
        entity_baseline = NamedEntityConsistencyBaseline()
        
        article = "John Smith works at the company. He was born in 1990."
        summary = "John Smith is an employee born in 1990."
        
        result = entity_baseline.evaluate_consistency(article, summary)
        
        assert result['consistency_score'] >= 0.0
        assert 'entity_scores' in result
    
    def test_evaluate_consistency_inconsistent(self):
        """Test consistency evaluation with inconsistencies"""
        entity_baseline = NamedEntityConsistencyBaseline()
        
        article = "John Smith works at Microsoft. He was born in 1990."
        summary = "Jane Doe works at Apple. She was born in 1985."
        
        result = entity_baseline.evaluate_consistency(article, summary)
        
        assert 0.0 <= result['consistency_score'] <= 1.0
        assert 'entity_scores' in result
    
    def test_evaluate_batch(self):
        """Test batch entity consistency evaluation"""
        entity_baseline = NamedEntityConsistencyBaseline()
        
        examples = [
            FactualityExample(
                article="John Smith works at the company.",
                summary="John Smith is an employee."
            ),
            FactualityExample(
                article="The company was founded in 1990.",
                summary="Founded in 1990."
            )
        ]
        
        results = entity_baseline.evaluate_batch(examples)
        
        assert len(results) == 2
        assert all('consistency_score' in result for result in results)


class TestSOTAMetricsEvaluator:
    """Test SOTA metrics evaluator"""
    
    def test_evaluator_initialization(self):
        """Test SOTA evaluator initialization"""
        evaluator = SOTAMetricsEvaluator()
        
        assert 'rouge' in evaluator.metrics
        assert 'bertscore' in evaluator.metrics
        assert 'factcheck' in evaluator.metrics
        assert 'entity_consistency' in evaluator.metrics
    
    def test_evaluator_selective_metrics(self):
        """Test evaluator with selective metrics"""
        evaluator = SOTAMetricsEvaluator(use_rouge=True, use_bertscore=False, 
                                       use_factcheck=True, use_entity_consistency=False)
        
        assert 'rouge' in evaluator.metrics
        assert 'bertscore' not in evaluator.metrics
        assert 'factcheck' in evaluator.metrics
        assert 'entity_consistency' not in evaluator.metrics
    
    def test_evaluate_example(self):
        """Test single example evaluation"""
        evaluator = SOTAMetricsEvaluator()
        
        example = FactualityExample(
            article="This is a test article with some content.",
            summary="This is a test summary."
        )
        
        results = evaluator.evaluate_example(example)
        
        assert 'rouge' in results
        assert 'bertscore' in results
        assert 'factcheck' in results
        assert 'entity_consistency' in results
    
    def test_evaluate_batch(self):
        """Test batch evaluation"""
        evaluator = SOTAMetricsEvaluator()
        
        examples = [
            FactualityExample(
                article="First article content.",
                summary="First summary."
            ),
            FactualityExample(
                article="Second article content.",
                summary="Second summary."
            )
        ]
        
        results = evaluator.evaluate_batch(examples)
        
        assert len(results) == 2
        assert all('rouge' in result for result in results)
    
    def test_aggregate_results(self):
        """Test result aggregation"""
        evaluator = SOTAMetricsEvaluator()
        
        # Mock batch results
        batch_results = [
            {
                'rouge': {'rouge1': {'precision': 0.8, 'recall': 0.7, 'fmeasure': 0.75}},
                'factcheck': {'factuality_score': 0.6}
            },
            {
                'rouge': {'rouge1': {'precision': 0.6, 'recall': 0.9, 'fmeasure': 0.72}},
                'factcheck': {'factuality_score': 0.8}
            }
        ]
        
        aggregated = evaluator.aggregate_results(batch_results)
        
        assert 'rouge' in aggregated
        assert 'factcheck' in aggregated
        assert aggregated['factcheck']['mean_factuality_score'] == 0.7


class TestBaselineComparator:
    """Test baseline comparison functionality"""
    
    def test_comparator_initialization(self):
        """Test baseline comparator initialization"""
        comparator = BaselineComparator()
        
        assert 'rouge' in comparator.baselines
        assert 'bertscore' in comparator.baselines
        assert 'factcheck' in comparator.baselines
        assert 'entity_consistency' in comparator.baselines
    
    def test_compare_baselines(self):
        """Test baseline comparison"""
        comparator = BaselineComparator()
        
        examples = [
            FactualityExample(
                article="This is a test article according to sources.",
                summary="This is a test summary."
            ),
            FactualityExample(
                article="Another test article with content.",
                summary="Another test summary."
            )
        ]
        
        results = comparator.compare_baselines(examples)
        
        assert 'rouge' in results
        assert 'bertscore' in results
        assert 'factcheck' in results
        assert 'entity_consistency' in results
        assert len(results['rouge']) == 2
    
    def test_compare_with_ground_truth(self):
        """Test baseline comparison with ground truth"""
        comparator = BaselineComparator()
        
        examples = [
            FactualityExample(
                article="Test article.",
                summary="Test summary."
            )
        ]
        ground_truth = [0.8]
        
        results = comparator.compare_baselines(examples, ground_truth)
        
        assert 'correlations' in results
        assert all(baseline in results['correlations'] 
                  for baseline in ['rouge', 'bertscore', 'factcheck', 'entity_consistency'])
    
    def test_generate_comparison_report(self):
        """Test comparison report generation"""
        comparator = BaselineComparator()
        
        # First run comparison
        examples = [
            FactualityExample(
                article="Test article.",
                summary="Test summary."
            )
        ]
        comparator.compare_baselines(examples)
        
        report = comparator.generate_comparison_report()
        
        assert isinstance(report, str)
        assert "Baseline Comparison Report" in report
        assert "ROUGE" in report.upper()


class TestBaselinesIntegration:
    """Test integration of baseline methods"""
    
    def test_full_baseline_pipeline(self):
        """Test complete baseline evaluation pipeline"""
        # Create test data
        examples = [
            FactualityExample(
                article="John Smith works at Microsoft Corp according to recent reports. The company was founded in 1975.",
                summary="John Smith is employed at Microsoft, which was founded in 1975.",
                dataset_name="test_dataset"
            ),
            FactualityExample(
                article="The study found significant results. Researchers said the data shows clear patterns.",
                summary="Research data shows significant findings according to the study.",
                dataset_name="test_dataset"
            )
        ]
        
        # Initialize evaluator
        evaluator = SOTAMetricsEvaluator()
        
        # Evaluate examples
        results = evaluator.evaluate_batch(examples)
        
        # Aggregate results
        aggregated = evaluator.aggregate_results(results)
        
        # Compare baselines
        comparator = BaselineComparator()
        comparison = comparator.compare_baselines(examples)
        
        # Generate report
        report = comparator.generate_comparison_report()
        
        # Verify pipeline completion
        assert len(results) == 2
        assert len(aggregated) > 0
        assert len(comparison) >= 4  # At least 4 baseline methods
        assert isinstance(report, str)
        
        # Verify all metrics were computed
        for result in results:
            assert 'rouge' in result
            assert 'bertscore' in result
            assert 'factcheck' in result
            assert 'entity_consistency' in result
    
    def test_baseline_error_handling(self):
        """Test baseline methods with invalid input"""
        evaluator = SOTAMetricsEvaluator()
        
        # Test with invalid example
        invalid_example = {"invalid": "format"}
        result = evaluator.evaluate_example(invalid_example)
        
        assert 'error' in result
        
        # Test with empty examples
        empty_results = evaluator.evaluate_batch([])
        assert len(empty_results) == 0
        
        # Test aggregation with empty results
        empty_aggregated = evaluator.aggregate_results([])
        assert len(empty_aggregated) == 0
