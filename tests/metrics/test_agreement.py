from utils import mock_environment, create_item

# Mock environment before imports
mock_environment()

# Now import the agreement functions
from consistent_agents.metrics.agreement_functions.bertscore import BERTScoreAgreement
from consistent_agents.metrics.agreement_functions.rouge import ROUGEAgreement
from consistent_agents.metrics.agreement_functions.llm_judge import LLMJudgeAgreement

import pytest


class TestBERTScoreAgreement:
    """Tests for BERTScore agreement function."""
    
    def test_call_no_exception(self):
        """Test that calling BERTScoreAgreement with two strings doesn't raise exceptions."""
        agreement_fn = BERTScoreAgreement()
        result = agreement_fn("This is output one", "This is output two")
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestROUGEAgreement:
    """Tests for ROUGE agreement function."""
    
    def test_call_no_exception(self):
        """Test that calling ROUGEAgreement with two strings doesn't raise exceptions."""
        agreement_fn = ROUGEAgreement()
        result = agreement_fn("This is output one", "This is output two")
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestLLMJudgeAgreement:
    """Tests for LLM Judge agreement function."""
    
    def test_call_no_exception(self):
        """Test that calling LLMJudgeAgreement with two strings doesn't raise exceptions."""
        agreement_fn = LLMJudgeAgreement()
        result = agreement_fn("This is output one", "This is output two", question="What is the answer?")
        
        assert isinstance(result, float)
        assert result in [0.0, 1.0]