"""
Essential Task Implementation Tests
=================================

Tests core logic for the three factuality tasks:
- Entailment Inference (binary)
- Summary Ranking 
- Consistency Rating (0-100)

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Union


class MockOpenAIClient:
    """Mock client for testing task implementations"""
    
    def __init__(self, response_map=None):
        self.response_map = response_map or {
            'entailment': "ENTAILMENT",
            'ranking': "1, 2, 3", 
            'rating': "85"
        }
        self.call_count = 0
    
    def complete(self, prompt, **kwargs):
        self.call_count += 1
        
        if 'entailment' in prompt.lower():
            content = self.response_map['entailment']
        elif 'rank' in prompt.lower():
            content = self.response_map['ranking']
        elif 'rate' in prompt.lower():
            content = self.response_map['rating']
        else:
            content = "ERROR"
            
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 100}
        }


class EntailmentInferenceTask:
    """Simplified entailment task for testing"""
    
    def __init__(self, client):
        self.client = client
    
    def evaluate(self, source: str, summary: str) -> dict:
        prompt = f"Determine if summary is entailed by source:\nSource: {source}\nSummary: {summary}\nAnswer: ENTAILMENT or CONTRADICTION"
        
        try:
            response = self.client.complete(prompt)
            content = response["choices"][0]["message"]["content"].strip().upper()
            
            if "ENTAILMENT" in content:
                prediction = 1
            elif "CONTRADICTION" in content:
                prediction = 0
            else:
                prediction = None
                
            return {
                "prediction": prediction,
                "success": prediction is not None,
                "raw_response": content
            }
        except Exception as e:
            return {
                "prediction": None,
                "success": False,
                "error_message": str(e)
            }


class SummaryRankingTask:
    """Simplified ranking task for testing"""
    
    def __init__(self, client):
        self.client = client
    
    def evaluate(self, source: str, summaries: List[str]) -> dict:
        summaries_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries)])
        prompt = f"Rank summaries by factual consistency:\nSource: {source}\nSummaries:\n{summaries_text}\nRanking (comma-separated):"
        
        try:
            response = self.client.complete(prompt)
            content = response["choices"][0]["message"]["content"].strip()
            
            # Parse ranking
            rankings = [int(x.strip()) for x in content.split(",")]
            
            # Validate ranking
            if len(rankings) == len(summaries) and set(rankings) == set(range(1, len(summaries) + 1)):
                prediction = rankings
            else:
                prediction = None
                
            return {
                "prediction": prediction,
                "success": prediction is not None,
                "raw_response": content
            }
        except Exception as e:
            return {
                "prediction": None,
                "success": False,
                "error_message": str(e)
            }


class ConsistencyRatingTask:
    """Simplified rating task for testing"""
    
    def __init__(self, client):
        self.client = client
    
    def evaluate(self, source: str, summary: str) -> dict:
        prompt = f"Rate factual consistency (0-100):\nSource: {source}\nSummary: {summary}\nRating:"
        
        try:
            response = self.client.complete(prompt)
            content = response["choices"][0]["message"]["content"].strip()
            
            rating = float(content)
            
            if 0 <= rating <= 100:
                prediction = rating
            else:
                prediction = None
                
            return {
                "prediction": prediction,
                "success": prediction is not None,
                "raw_response": content
            }
        except Exception as e:
            return {
                "prediction": None,
                "success": False,
                "error_message": str(e)
            }


class TestEntailmentInferenceTask:
    """Test entailment inference task"""
    
    @pytest.fixture
    def task(self):
        client = MockOpenAIClient({'entailment': "ENTAILMENT"})
        return EntailmentInferenceTask(client)
    
    def test_entailment_prediction(self, task):
        """Test positive entailment prediction"""
        result = task.evaluate(
            source="The cat sat on the mat.",
            summary="A cat was on the mat."
        )
        
        assert result["success"] is True
        assert result["prediction"] == 1
        assert "ENTAILMENT" in result["raw_response"]
    
    def test_contradiction_prediction(self):
        """Test contradiction prediction"""
        client = MockOpenAIClient({'entailment': "CONTRADICTION"})
        task = EntailmentInferenceTask(client)
        
        result = task.evaluate(
            source="The cat sat on the mat.",
            summary="The dog was in the yard."
        )
        
        assert result["success"] is True
        assert result["prediction"] == 0
        assert "CONTRADICTION" in result["raw_response"]
    
    def test_invalid_response(self):
        """Test handling of invalid API response"""
        client = MockOpenAIClient({'entailment': "INVALID_RESPONSE"})
        task = EntailmentInferenceTask(client)
        
        result = task.evaluate(
            source="Source text",
            summary="Summary text"
        )
        
        assert result["success"] is False
        assert result["prediction"] is None


class TestSummaryRankingTask:
    """Test summary ranking task"""
    
    @pytest.fixture
    def task(self):
        client = MockOpenAIClient({'ranking': "1, 2, 3"})
        return SummaryRankingTask(client)
    
    def test_valid_ranking(self, task):
        """Test valid ranking prediction"""
        summaries = [
            "High quality summary",
            "Medium quality summary", 
            "Low quality summary"
        ]
        
        result = task.evaluate(
            source="Source document text",
            summaries=summaries
        )
        
        assert result["success"] is True
        assert result["prediction"] == [1, 2, 3]
        assert len(result["prediction"]) == len(summaries)
    
    def test_invalid_ranking_format(self):
        """Test handling of invalid ranking format"""
        client = MockOpenAIClient({'ranking': "invalid ranking"})
        task = SummaryRankingTask(client)
        
        result = task.evaluate(
            source="Source text",
            summaries=["Summary 1", "Summary 2"]
        )
        
        assert result["success"] is False
        assert result["prediction"] is None
    
    def test_wrong_ranking_length(self):
        """Test handling of wrong number of rankings"""
        client = MockOpenAIClient({'ranking': "1, 2"})  # Only 2 rankings for 3 summaries
        task = SummaryRankingTask(client)
        
        result = task.evaluate(
            source="Source text",
            summaries=["Summary 1", "Summary 2", "Summary 3"]
        )
        
        assert result["success"] is False
        assert result["prediction"] is None


class TestConsistencyRatingTask:
    """Test consistency rating task"""
    
    @pytest.fixture 
    def task(self):
        client = MockOpenAIClient({'rating': "85"})
        return ConsistencyRatingTask(client)
    
    def test_valid_rating(self, task):
        """Test valid rating prediction"""
        result = task.evaluate(
            source="The cat sat on the mat.",
            summary="A cat was sitting on a mat."
        )
        
        assert result["success"] is True
        assert result["prediction"] == 85.0
        assert 0 <= result["prediction"] <= 100
    
    def test_boundary_ratings(self):
        """Test boundary rating values"""
        # Test minimum rating
        client = MockOpenAIClient({'rating': "0"})
        task = ConsistencyRatingTask(client)
        result = task.evaluate("Source", "Summary")
        assert result["prediction"] == 0.0
        
        # Test maximum rating  
        client = MockOpenAIClient({'rating': "100"})
        task = ConsistencyRatingTask(client)
        result = task.evaluate("Source", "Summary")
        assert result["prediction"] == 100.0
    
    def test_invalid_rating_range(self):
        """Test handling of out-of-range ratings"""
        client = MockOpenAIClient({'rating': "150"})  # Above 100
        task = ConsistencyRatingTask(client)
        
        result = task.evaluate("Source", "Summary")
        
        assert result["success"] is False
        assert result["prediction"] is None
    
    def test_non_numeric_rating(self):
        """Test handling of non-numeric rating"""
        client = MockOpenAIClient({'rating': "not a number"})
        task = ConsistencyRatingTask(client)
        
        result = task.evaluate("Source", "Summary")
        
        assert result["success"] is False
        assert result["prediction"] is None