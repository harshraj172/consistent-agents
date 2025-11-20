from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the agreement functions
from consistent_agents.metrics.agreement_functions.bertscore import bertscore
from consistent_agents.metrics.agreement_functions.rouge import rouge
from consistent_agents.metrics.agreement_functions.consistency import consistency
from consistent_agents.metrics.agreement_functions.entailment import entailment
from consistent_agents.metrics.agreement_functions.contradiction import contradiction

import pytest
from unittest.mock import patch, MagicMock


class TestBERTScoreAgreement:
    """Tests for BERTScore agreement function."""
    
    @patch('bert_score.BERTScorer')
    def test_call_no_exception(self, mock_scorer_class):
        """Test that calling bertscore with two strings doesn't raise exceptions."""
        # Mock the scorer
        mock_scorer = MagicMock()
        mock_f1 = MagicMock()
        mock_f1.item.return_value = 0.85
        mock_scorer.score.return_value = (MagicMock(), MagicMock(), mock_f1)
        mock_scorer_class.return_value = mock_scorer
        
        result = bertscore("This is output one", "This is output two")
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestROUGEAgreement:
    """Tests for ROUGE agreement function."""
    
    @patch('rouge_score.rouge_scorer.RougeScorer')
    def test_call_no_exception(self, mock_rouge_scorer):
        """Test that calling rouge with two strings doesn't raise exceptions."""
        # Mock the scorer
        mock_scorer = MagicMock()
        mock_score = MagicMock()
        mock_score.fmeasure = 0.75
        mock_scorer.score.return_value = {"rougeL": mock_score}
        mock_rouge_scorer.return_value = mock_scorer
        
        result = rouge("This is output one", "This is output two")
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestConsistencyAgreement:
    """Tests for Consistency agreement function."""
    
    @patch('openai.OpenAI')
    @patch('consistent_agents.metrics.agreement_functions.consistency.Path')
    def test_call_consistent_response(self, mock_path, mock_openai):
        """Test that consistency returns 1.0 when LLM judges outputs as consistent."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {question} {prediction1} {prediction2}"
        mock_path.return_value = mock_template_file
        
        result = consistency(
            "Output one", 
            "Output one", 
            question="What is the answer?"
        )
        
        assert isinstance(result, float)
        assert result == 1.0
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('openai.OpenAI')
    @patch('consistent_agents.metrics.agreement_functions.consistency.Path')
    def test_call_inconsistent_response(self, mock_path, mock_openai):
        """Test that consistency returns 0.0 when LLM judges outputs as inconsistent."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {question} {prediction1} {prediction2}"
        mock_path.return_value = mock_template_file
        
        result = consistency(
            "Output one", 
            "Output two", 
            question="What is the answer?"
        )
        
        assert isinstance(result, float)
        assert result == 0.0


class TestEntailmentAgreement:
    """Tests for Entailment agreement function."""
    
    @patch('openai.OpenAI')
    @patch('consistent_agents.metrics.agreement_functions.entailment.Path')
    def test_call_entailed_response(self, mock_path, mock_openai):
        """Test that entailment returns 1.0 when LLM judges outputs as mutually entailed."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {sentence_a} {sentence_b}"
        mock_path.return_value = mock_template_file
        
        result = entailment("Sentence A", "Sentence A")
        
        assert isinstance(result, float)
        assert result == 1.0
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('openai.OpenAI')
    @patch('consistent_agents.metrics.agreement_functions.entailment.Path')
    def test_call_not_entailed_response(self, mock_path, mock_openai):
        """Test that entailment returns 0.0 when LLM judges outputs as not mutually entailed."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {sentence_a} {sentence_b}"
        mock_path.return_value = mock_template_file
        
        result = entailment("Sentence A", "Sentence B")
        
        assert isinstance(result, float)
        assert result == 0.0


class TestContradictionAgreement:
    """Tests for Contradiction agreement function."""
    
    @patch('openai.OpenAI')
    @patch('consistent_agents.metrics.agreement_functions.contradiction.Path')
    def test_call_no_contradiction_response(self, mock_path, mock_openai):
        """Test that contradiction returns 1.0 when LLM judges no contradiction."""
        # Mock OpenAI client - "No" means no contradiction, so return 1.0
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {sentence_a} {sentence_b}"
        mock_path.return_value = mock_template_file
        
        result = contradiction("Sentence A", "Sentence A")
        
        assert isinstance(result, float)
        assert result == 1.0
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('openai.OpenAI')
    @patch('consistent_agents.metrics.agreement_functions.contradiction.Path')
    def test_call_contradiction_response(self, mock_path, mock_openai):
        """Test that contradiction returns 0.0 when LLM judges contradiction exists."""
        # Mock OpenAI client - "Yes" means contradiction, so return 0.0
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {sentence_a} {sentence_b}"
        mock_path.return_value = mock_template_file
        
        result = contradiction("Sentence A", "Sentence B")
        
        assert isinstance(result, float)
        assert result == 0.0