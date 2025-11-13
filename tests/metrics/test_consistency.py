from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the consistency metric
from consistent_agents.metrics.consistency import ConsistencyMetric

import pytest
from unittest.mock import MagicMock


class TestConsistencyMetric:
    """Tests for Consistency metric."""
    
    def test_item_score_with_mocked_components(self):
        """Test that item_score correctly uses agreement function and aggregator."""
        # Create mock agreement function
        mock_agreement_fn = MagicMock()
        mock_agreement_fn.__class__.__name__ = "MockAgreement"
        mock_agreement_fn.return_value = 0.8
        
        # Create mock aggregator
        mock_aggregator = MagicMock()
        mock_aggregator.__class__.__name__ = "MockAggregator"
        mock_aggregator.return_value = 0.75
        
        # Create consistency metric with mocked components
        metric = ConsistencyMetric()
        metric.agreement_function = mock_agreement_fn
        metric.aggregator = mock_aggregator
        
        item = create_item({"question": "What is 2+2?"})
        perturbed_outputs = [
            {"output": "The answer is 4"},
            {"output": "Four"},
            {"output": "4"}
        ]
        
        score = metric.item_score(item, perturbed_outputs)
        
        # Check that aggregator was called with correct arguments
        mock_aggregator.assert_called_once()
        call_args = mock_aggregator.call_args
        assert call_args[0][0] == ["The answer is 4", "Four", "4"]  # outputs
        assert call_args[0][1] == mock_agreement_fn  # agreement function
        assert call_args[1]['question'] == "What is 2+2?"
        
        # Check that score was stored
        assert len(metric.scores) == 1
        assert metric.scores[0] == 0.75
        assert score == 0.75
    
    def test_total_score_calculation(self):
        """Test that total_score calculates average correctly."""
        metric = ConsistencyMetric()
        metric.agreement_function = MagicMock()
        metric.aggregator = MagicMock(side_effect=[0.8, 0.6, 0.9])  # Different scores for each call
        
        item1 = create_item({"question": "Q1"})
        item2 = create_item({"question": "Q2"})
        item3 = create_item({"question": "Q3"})
        
        metric.item_score(item1, [{"output": "A1"}])
        metric.item_score(item2, [{"output": "A2"}])
        metric.item_score(item3, [{"output": "A3"}])
        
        # Total score should be average: (0.8 + 0.6 + 0.9) / 3 = 0.766...
        total = metric.total_score()
        assert total == pytest.approx(0.7666666666666666, abs=0.01)
        assert len(metric.scores) == 3
    
    def test_total_score_empty(self):
        """Test that total_score returns 0.0 when no items processed."""
        metric = ConsistencyMetric()
        metric.agreement_function = MagicMock()
        metric.aggregator = MagicMock()
        
        assert metric.total_score() == 0.0
        assert len(metric.scores) == 0
    
    def test_item_score_extracts_outputs_correctly(self):
        """Test that item_score correctly extracts outputs from perturbed_outputs."""
        metric = ConsistencyMetric()
        
        # Track what outputs are passed to aggregator
        captured_outputs = []
        def capture_outputs(outputs, agreement_fn, question=None, **kwargs):
            captured_outputs.append(outputs)
            return 0.5
        
        metric.agreement_function = MagicMock()
        metric.aggregator = MagicMock(side_effect=capture_outputs)
        
        item = create_item({"question": "Test question"})
        perturbed_outputs = [
            {"output": "Output 1", "type": "paraphrase"},
            {"output": "Output 2", "type": "translation"},
            {"output": "Output 3", "type": "noise"}
        ]
        
        metric.item_score(item, perturbed_outputs)
        
        # Verify outputs were extracted correctly
        assert len(captured_outputs) == 1
        assert captured_outputs[0] == ["Output 1", "Output 2", "Output 3"]