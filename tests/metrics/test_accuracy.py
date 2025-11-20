import pytest
from unittest.mock import patch, MagicMock
from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the accuracy metric function
from consistent_agents.metrics.accuracy import score as accuracy_score, _judge_accuracy


class TestAccuracyMetric:
    """Tests for Accuracy metric function."""
    
    @patch('consistent_agents.metrics.accuracy._judge_accuracy')
    @patch('consistent_agents.metrics.accuracy.Path')
    def test_score_with_mocked_judge(self, mock_path, mock_judge):
        """Test that score correctly counts correct predictions."""
        # Mock prompt template file
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {question} {prediction} {correct_answers} {incorrect_answers}"
        mock_path.return_value = mock_template_file
        
        # Mock the judge to return: 2 correct, 1 incorrect
        mock_judge.side_effect = [True, False, True]
        
        question = "What is 2+2?"
        predictions = ["output1", "output2", "output3"]
        correct_answers = ["4", "four"]
        incorrect_answers = ["5", "five"]
        
        correct_count, total_predictions = accuracy_score(
            question=question,
            predictions=predictions,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers
        )
        
        # Should be 2 correct out of 3
        assert correct_count == 2
        assert total_predictions == 3
        assert mock_judge.call_count == 3
    
    @patch('consistent_agents.metrics.accuracy._judge_accuracy')
    @patch('consistent_agents.metrics.accuracy.Path')
    def test_all_correct_predictions(self, mock_path, mock_judge):
        """Test when all predictions are correct."""
        # Mock prompt template file
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {question} {prediction} {correct_answers} {incorrect_answers}"
        mock_path.return_value = mock_template_file
        
        mock_judge.return_value = True
        
        question = "What is 2+2?"
        predictions = ["output1", "output2", "output3"]
        correct_answers = ["4"]
        incorrect_answers = ["5"]
        
        correct_count, total_predictions = accuracy_score(
            question=question,
            predictions=predictions,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers
        )
        
        assert correct_count == 3
        assert total_predictions == 3
    
    @patch('consistent_agents.metrics.accuracy._judge_accuracy')
    @patch('consistent_agents.metrics.accuracy.Path')
    def test_all_incorrect_predictions(self, mock_path, mock_judge):
        """Test when all predictions are incorrect."""
        # Mock prompt template file
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {question} {prediction} {correct_answers} {incorrect_answers}"
        mock_path.return_value = mock_template_file
        
        mock_judge.return_value = False
        
        question = "What is 2+2?"
        predictions = ["output1", "output2", "output3"]
        correct_answers = ["4"]
        incorrect_answers = ["5"]
        
        correct_count, total_predictions = accuracy_score(
            question=question,
            predictions=predictions,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers
        )
        
        assert correct_count == 0
        assert total_predictions == 3
    
    @patch('openai.OpenAI')
    def test_judge_accuracy_with_openai_mock(self, mock_openai):
        """Test _judge_accuracy function with OpenAI mocking."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = _judge_accuracy(
            question="What is 2+2?",
            prediction="4",
            correct_answers=["4", "four"],
            incorrect_answers=["5", "five"],
            prompt_template="Template: {question} {prediction} {correct_answers} {incorrect_answers}",
            judge_model="gpt-4o-mini"
        )
        
        assert result is True
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_judge_accuracy_returns_false_for_no(self, mock_openai):
        """Test _judge_accuracy returns False when LLM says No."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = _judge_accuracy(
            question="What is 2+2?",
            prediction="5",
            correct_answers=["4", "four"],
            incorrect_answers=["5", "five"],
            prompt_template="Template: {question} {prediction} {correct_answers} {incorrect_answers}",
            judge_model="gpt-4o-mini"
        )
        
        assert result is False
    
    @patch('openai.OpenAI')
    def test_judge_accuracy_handles_exceptions(self, mock_openai):
        """Test _judge_accuracy handles exceptions gracefully."""
        # Mock OpenAI client to raise exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        result = _judge_accuracy(
            question="What is 2+2?",
            prediction="4",
            correct_answers=["4"],
            incorrect_answers=["5"],
            prompt_template="Template: {question} {prediction} {correct_answers} {incorrect_answers}",
            judge_model="gpt-4o-mini"
        )
        
        assert result is False
