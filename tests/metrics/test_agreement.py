from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the agreement functions
from consistent_agents.metrics.agreement_functions.bertscore import bertscore
from consistent_agents.metrics.agreement_functions.rouge import rouge
from consistent_agents.metrics.agreement_functions.llm_as_judge import llm_as_judge
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


class TestLLMAsJudgeAgreement:
    """Tests for LLM as Judge agreement function."""
    
    @patch('litellm.completion')
    @patch('consistent_agents.metrics.agreement_functions.llm_as_judge.Path')
    def test_call_consistent_response(self, mock_path, mock_litellm):
        """Test that llm_as_judge returns 1.0 when LLM judges outputs as consistent."""
        # Mock litellm completion
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes"
        mock_litellm.return_value = mock_response
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {question} {prediction1} {prediction2}"
        mock_path.return_value = mock_template_file
        
        from consistent_agents.metrics.agreement_functions.llm_as_judge import llm_as_judge
        
        result = llm_as_judge(
            "Output one", 
            "Output one", 
            question="What is the answer?",
            prompt_template_path=mock_template_file
        )
        
        assert isinstance(result, float)
        assert result == 1.0
        mock_litellm.assert_called_once()
    
    @patch('litellm.completion')
    @patch('consistent_agents.metrics.agreement_functions.llm_as_judge.Path')
    def test_call_inconsistent_response(self, mock_path, mock_litellm):
        """Test that llm_as_judge returns 0.0 when LLM judges outputs as inconsistent."""
        # Mock litellm completion
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        mock_litellm.return_value = mock_response
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {question} {prediction1} {prediction2}"
        mock_path.return_value = mock_template_file
        
        from consistent_agents.metrics.agreement_functions.llm_as_judge import llm_as_judge
        
        result = llm_as_judge(
            "Output one", 
            "Output two", 
            question="What is the answer?",
            prompt_template_path=mock_template_file
        )
        
        assert isinstance(result, float)
        assert result == 0.0


class TestEntailmentAgreement:
    """Tests for Entailment agreement function."""
    
    @patch('consistent_agents.models.hfmodel.get_huggingface_model')
    def test_call_entailed_response(self, mock_get_model):
        """Test that entailment returns 1.0 when BERT judges outputs as mutually entailed."""
        # Mock the HuggingFace model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_model.tokenizer = mock_tokenizer
        
        # Mock model outputs
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_softmax = MagicMock()
        # Create a tensor-like object that supports .T[2].item()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 1.0
        mock_softmax.T = [MagicMock(), MagicMock(), mock_tensor]
        mock_logits.softmax.return_value = mock_softmax
        mock_outputs.logits = mock_logits
        mock_model.model.return_value = mock_outputs
        
        mock_get_model.return_value = mock_model
        
        # Patch the model to use tokenizer instead of detection_tokenizer
        with patch.object(mock_model, 'detection_tokenizer', mock_tokenizer, create=True), \
             patch.object(mock_model, 'detection_model', mock_model.model, create=True):
            result = entailment("Sentence A", "Sentence A")
        
        assert isinstance(result, float)
        assert round(result) == 1
    
    @patch('litellm.completion')
    @patch('consistent_agents.metrics.agreement_functions.entailment.Path')
    def test_call_not_entailed_response(self, mock_path, mock_litellm):
        """Test that entailment returns 0.0 when LLM judges outputs as not mutually entailed."""
        # Mock litellm completion
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        mock_litellm.return_value = mock_response
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {sentence_a} {sentence_b}"
        mock_path.return_value = mock_template_file
        
        result = entailment(
            "Sentence A", 
            "Sentence B",
            judge_model="gpt-4o-mini",
            prompt_template_path=mock_template_file
        )
        
        assert isinstance(result, float)
        assert result == 0.0


class TestContradictionAgreement:
    """Tests for Contradiction agreement function."""
    
    @patch('consistent_agents.models.hfmodel.get_huggingface_model')
    def test_call_no_contradiction_response(self, mock_get_model):
        """Test that contradiction returns 1.0 when BERT judges no contradiction."""
        # Mock the HuggingFace model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_model.tokenizer = mock_tokenizer
        
        # Mock model outputs - no contradiction (score should be high)
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_softmax = MagicMock()
        # Create a tensor-like object that supports .T[0].item()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 1.0  # High score means no contradiction
        mock_softmax.T = [mock_tensor, MagicMock(), MagicMock()]
        mock_logits.softmax.return_value = mock_softmax
        mock_outputs.logits = mock_logits
        mock_model.model.return_value = mock_outputs
        
        mock_get_model.return_value = mock_model
        
        # Patch the model to use tokenizer instead of detection_tokenizer
        with patch.object(mock_model, 'detection_tokenizer', mock_tokenizer, create=True), \
             patch.object(mock_model, 'detection_model', mock_model.model, create=True):
            result = contradiction("Sentence A", "Sentence A")
        
        assert isinstance(result, float)
        assert round(result) == 0
    
    @patch('litellm.completion')
    @patch('consistent_agents.metrics.agreement_functions.contradiction.Path')
    def test_call_contradiction_response(self, mock_path, mock_litellm):
        """Test that contradiction returns 0.0 when LLM judges contradiction exists."""
        # Mock litellm completion - "Yes" means contradiction, so return 0.0
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes"
        mock_litellm.return_value = mock_response
        
        # Mock prompt template path
        mock_template_file = MagicMock()
        mock_template_file.read_text.return_value = "Template: {sentence_a} {sentence_b}"
        mock_path.return_value = mock_template_file
        
        result = contradiction(
            "Sentence A", 
            "Sentence B",
            judge_model="gpt-4o-mini",
            prompt_template_path=mock_template_file
        )
        
        assert isinstance(result, float)
        assert result == 0.0