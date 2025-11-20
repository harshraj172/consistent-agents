from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the consistency metric function
from consistent_agents.metrics.consistency import score as consistency_score
import consistent_agents.metrics.consistency as consistency_module

import pytest
from unittest.mock import patch, MagicMock


class TestConsistencyMetric:
    """Tests for Consistency metric function."""
    
    @patch('consistent_agents.metrics.aggregators.pairwise.pairwise')
    @patch('consistent_agents.metrics.agreement_functions.consistency.consistency')
    def test_score_with_mocked_components(self, mock_consistency_fn, mock_pairwise_fn):
        """Test that score correctly uses agreement function and aggregator."""
        # Replace in dictionary
        original_pairwise = consistency_module.AGGREGATORS['pairwise']
        consistency_module.AGGREGATORS['pairwise'] = mock_pairwise_fn
        
        try:
            # Mock the agreement function to return 0.8
            mock_consistency_fn.return_value = 0.8
            
            # Mock the aggregator to return (2.4, 3) - total agreement and pairs
            mock_pairwise_fn.return_value = (2.4, 3)
            
            outputs = ["Output 1", "Output 2", "Output 3"]
            result = consistency_score(outputs, question="What is 2+2?")
            
            # Check that aggregator was called
            mock_pairwise_fn.assert_called_once()
            call_args = mock_pairwise_fn.call_args
            assert call_args[0][0] == outputs  # outputs
            
            # Result should be the aggregator's return value
            assert result == (2.4, 3)
        finally:
            # Restore original
            consistency_module.AGGREGATORS['pairwise'] = original_pairwise
    
    @patch('consistent_agents.metrics.aggregators.entropy.entropy')
    @patch('consistent_agents.metrics.agreement_functions.bertscore.bertscore')
    def test_score_with_entropy_aggregator(self, mock_bertscore_fn, mock_entropy_fn):
        """Test that score works with entropy aggregator."""
        # Replace in dictionary
        original_entropy = consistency_module.AGGREGATORS['entropy']
        consistency_module.AGGREGATORS['entropy'] = mock_entropy_fn
        
        try:
            # Mock the agreement function
            mock_bertscore_fn.return_value = 0.7
            
            # Mock the entropy aggregator to return entropy value
            mock_entropy_fn.return_value = 1.5
            
            outputs = ["Output 1", "Output 2", "Output 3"]
            result = consistency_score(
                outputs, 
                question="Test question",
                agreement="bertscore",
                aggregator="entropy"
            )
            
            # Check that entropy was called
            mock_entropy_fn.assert_called_once()
            assert result == 1.5
        finally:
            # Restore original
            consistency_module.AGGREGATORS['entropy'] = original_entropy
    
    @patch('consistent_agents.metrics.aggregators.pairwise.pairwise')
    @patch('consistent_agents.metrics.agreement_functions.rouge.rouge')
    def test_score_with_rouge_agreement(self, mock_rouge_fn, mock_pairwise_fn):
        """Test that score works with rouge agreement function."""
        # Replace in dictionary
        original_pairwise = consistency_module.AGGREGATORS['pairwise']
        consistency_module.AGGREGATORS['pairwise'] = mock_pairwise_fn
        
        try:
            # Mock the agreement function
            mock_rouge_fn.return_value = 0.85
            
            # Mock the aggregator
            mock_pairwise_fn.return_value = (2.55, 3)
            
            outputs = ["Output 1", "Output 2", "Output 3"]
            result = consistency_score(
                outputs,
                question="Test question",
                agreement="rouge",
                aggregator="pairwise"
            )
            
            # Check that pairwise was called
            mock_pairwise_fn.assert_called_once()
            assert result == (2.55, 3)
        finally:
            # Restore original
            consistency_module.AGGREGATORS['pairwise'] = original_pairwise
    
    def test_score_invalid_agreement(self):
        """Test that score raises error for invalid agreement function."""
        outputs = ["Output 1", "Output 2"]
        
        with pytest.raises(ValueError, match="Unknown agreement function"):
            consistency_score(outputs, agreement="invalid_agreement")
    
    def test_score_invalid_aggregator(self):
        """Test that score raises error for invalid aggregator."""
        outputs = ["Output 1", "Output 2"]
        
        with pytest.raises(ValueError, match="Unknown aggregator"):
            consistency_score(outputs, aggregator="invalid_aggregator")
    
    @patch('consistent_agents.metrics.aggregators.pairwise.pairwise')
    @patch('consistent_agents.metrics.agreement_functions.consistency.consistency')
    def test_score_passes_agreement_params(self, mock_consistency_fn, mock_pairwise_fn):
        """Test that score passes agreement_params correctly."""
        # Replace in dictionary
        original_pairwise = consistency_module.AGGREGATORS['pairwise']
        consistency_module.AGGREGATORS['pairwise'] = mock_pairwise_fn
        
        try:
            mock_consistency_fn.return_value = 0.8
            mock_pairwise_fn.return_value = (2.4, 3)
            
            outputs = ["Output 1", "Output 2", "Output 3"]
            agreement_params = {"judge_model": "gpt-4", "temperature": 0.2}
            
            consistency_score(
                outputs,
                question="Test question",
                agreement="consistency",
                agreement_params=agreement_params
            )
            
            # Verify that the agreement function wrapper was called with the right params
            # The wrapper should pass agreement_params to the actual function
            assert mock_pairwise_fn.called
        finally:
            # Restore original
            consistency_module.AGGREGATORS['pairwise'] = original_pairwise