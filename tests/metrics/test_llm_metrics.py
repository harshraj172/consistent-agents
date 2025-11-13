import pytest
from unittest.mock import patch
from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the metrics after mocking the environment
from consistent_agents.metrics.accuracy import AccuracyMetric
from consistent_agents.metrics.entailment import EntailmentMetric
from consistent_agents.metrics.contradiction import ContradictionMetric


# Parametrize over all three metric classes
@pytest.fixture(params=[
    (AccuracyMetric, '_judge_accuracy', create_item({
        "question": "What is 2+2?",
        "correct_answers": ["4", "four"],
        "incorrect_answers": ["5", "five"]
    })),
    (EntailmentMetric, '_judge_entailment', create_item({
        "base_output": "The answer is correct"
    })),
    (ContradictionMetric, '_judge_contradiction', create_item({
        "base_output": "Original answer"
    })),
])
def metric_setup(request):
    """Fixture that provides metric class, judge method name, and sample item."""
    MetricClass, judge_method, item = request.param
    return MetricClass(), judge_method, item


class TestLLMMetrics:
    def test_item_score_with_mocked_judge(self, metric_setup):
        """Test that item_score correctly counts correct predictions."""
        metric, judge_method, item = metric_setup
        
        # Mock the judge to return: 2 correct, 1 incorrect
        with patch.object(metric, judge_method, side_effect=[True, False, True]):
            perturbed_outputs = [
                {"output": "output1"},
                {"output": "output2"},
                {"output": "output3"}
            ]
            
            score = metric.item_score(item, perturbed_outputs)
            
            # Should be 2/3 = 0.666...
            assert score == pytest.approx(2/3)
            assert metric.total_correct == 2
            assert metric.total_comparisons == 3
    
    def test_total_score_calculation(self, metric_setup):
        """Test that total_score calculates correctly from accumulated counts."""
        metric, judge_method, item = metric_setup
        
        # Mock judge returns: True, False, True, False, True (3 correct out of 5)
        with patch.object(metric, judge_method, side_effect=[True, False, True, False, True]):
            # Create new items for each call
            item1 = create_item({k: getattr(item, k) for k in ['question', 'base_output', 'correct_answers', 'incorrect_answers'] if hasattr(item, k)})
            item2 = create_item({k: getattr(item, k) for k in ['question', 'base_output', 'correct_answers', 'incorrect_answers'] if hasattr(item, k)})
            
            metric.item_score(item1, [{"output": "a"}, {"output": "b"}])
            metric.item_score(item2, [{"output": "c"}, {"output": "d"}, {"output": "e"}])
            
            # 3 correct out of 5 total
            assert metric.total_score() == pytest.approx(3/5)
            assert metric.total_correct == 3
            assert metric.total_comparisons == 5
    
    def test_all_correct_predictions(self, metric_setup):
        """Test when all predictions are correct."""
        metric, judge_method, item = metric_setup
        
        with patch.object(metric, judge_method, return_value=True):
            perturbed_outputs = [
                {"output": "output1"},
                {"output": "output2"},
                {"output": "output3"}
            ]
            
            score = metric.item_score(item, perturbed_outputs)
            
            assert score == pytest.approx(1.0)
            assert metric.total_correct == 3
            assert metric.total_comparisons == 3
    
    def test_all_incorrect_predictions(self, metric_setup):
        """Test when all predictions are incorrect."""
        metric, judge_method, item = metric_setup
        
        with patch.object(metric, judge_method, return_value=False):
            perturbed_outputs = [
                {"output": "output1"},
                {"output": "output2"},
                {"output": "output3"}
            ]
            
            score = metric.item_score(item, perturbed_outputs)
            
            assert score == pytest.approx(0.0)
            assert metric.total_correct == 0
            assert metric.total_comparisons == 3