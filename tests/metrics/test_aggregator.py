from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the aggregators
from consistent_agents.metrics.aggregators.pairwise import PairwiseAggregator
from consistent_agents.metrics.aggregators.entropy import EntropyAggregator

import pytest


class MockAgreementFunction:
    """Mock agreement function for testing."""
    def __init__(self, agreement_scores=None):
        """
        Args:
            agreement_scores: Dict mapping (output1, output2) tuples to scores,
                             or a function that takes (output1, output2) and returns score
        """
        if agreement_scores is None:
            # Default: same strings = 1.0, different = 0.0
            self.agreement_scores = {}
        elif isinstance(agreement_scores, dict):
            self.agreement_scores = agreement_scores
        else:
            # Assume it's a callable
            self._score_func = agreement_scores
            self.agreement_scores = {}
    
    def __call__(self, output1: str, output2: str, question=None, **kwargs) -> float:
        if hasattr(self, '_score_func'):
            return self._score_func(output1, output2)
        
        # Check if we have a stored score
        key = (output1, output2)
        if key in self.agreement_scores:
            return self.agreement_scores[key]
        
        # Default behavior: same strings = 1.0, different = 0.0
        if output1 == output2:
            return 1.0
        return 0.0


class TestPairwiseAggregator:
    """Tests for Pairwise aggregator."""
    
    def test_pairwise_aggregator(self):
        """Test that PairwiseAggregator computes average agreement correctly."""
        aggregator = PairwiseAggregator()
        
        # Mock agreement function: same strings = 1.0, different = 0.5
        def agreement_fn(output1, output2, question=None, **kwargs):
            if output1 == output2:
                return 1.0
            return 0.5
        
        mock_agreement = MockAgreementFunction(agreement_fn)
        
        # Test with 3 outputs: ["A", "A", "B"]
        # Pairs: (A, A) = 1.0, (A, B) = 0.5, (A, B) = 0.5
        # Average = (1.0 + 0.5 + 0.5) / 3 = 2.0 / 3 ≈ 0.667
        outputs = ["Output A", "Output A", "Output B"]
        result = aggregator(outputs, mock_agreement)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(2.0 / 3.0)
    
    def test_pairwise_aggregator_single_output(self):
        """Test PairwiseAggregator with single output returns 1.0."""
        aggregator = PairwiseAggregator()
        mock_agreement = MockAgreementFunction()
        
        outputs = ["Single output"]
        result = aggregator(outputs, mock_agreement)
        
        assert result == 1.0


class TestEntropyAggregator:
    """Tests for Entropy aggregator."""
    
    def test_entropy_one_sentence_repeated_cluster_number_one(self):
        """Test that one sentence repeated multiple times results in 1 cluster."""
        aggregator = EntropyAggregator(threshold=0.5)
        
        # Mock agreement function: same strings = 1.0 (above threshold)
        def agreement_fn(output1, output2, question=None, **kwargs):
            if output1 == output2:
                return 1.0
            return 0.0
        
        mock_agreement = MockAgreementFunction(agreement_fn)
        
        # Same sentence repeated 5 times
        outputs = ["Same sentence", "Same sentence", "Same sentence", "Same sentence", "Same sentence"]
        
        # Get clusters by accessing the private method
        clusters = aggregator._semantic_clustering(outputs, mock_agreement, question=None, threshold=0.5)
        
        # Should have exactly 1 cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 5
    
    def test_entropy_two_sentences_alternating_cluster_number_two(self):
        """Test that two sentences repeated alternately results in 2 clusters."""
        aggregator = EntropyAggregator(threshold=0.5)
        
        # Mock agreement function: same strings = 1.0, different = 0.0
        def agreement_fn(output1, output2, question=None, **kwargs):
            if output1 == output2:
                return 1.0
            return 0.0
        
        mock_agreement = MockAgreementFunction(agreement_fn)
        
        # Two different sentences alternating
        outputs = [
            "First sentence",
            "Second sentence",
            "First sentence",
            "Second sentence",
            "First sentence",
            "Second sentence"
        ]
        
        # Get clusters by accessing the private method
        clusters = aggregator._semantic_clustering(outputs, mock_agreement, question=None, threshold=0.5)
        
        # Should have exactly 2 clusters
        assert len(clusters) == 2
        
        # Check that each cluster contains the correct sentences
        cluster_texts = [set(cluster) for cluster in clusters]
        assert {"First sentence"} in cluster_texts
        assert {"Second sentence"} in cluster_texts
        
        # Verify cluster sizes (3 of each)
        cluster_sizes = [len(c) for c in clusters]
        assert sorted(cluster_sizes) == [3, 3]