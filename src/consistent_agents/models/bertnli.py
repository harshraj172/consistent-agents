import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

__all__ = ["get_bert_nli_model", "entailment_bert", "contradiction_bert", "entailment_openai", "contradiction_openai"]

# Global cached model instance
_BERT_NLI_SINGLETON = None


class BERTNLI:
    """
    Allows for determining if two sentences contradict
    each other or if one sentence entails the other.

    Parameters:
        tok_path (str): Path to the tokenizer.
        model_path (str): Path to the model.
        max_length (int): Maximum length of the input sequences.
    """

    def __init__(
        self,
        tok_path="microsoft/deberta-base-mnli",
        model_path="microsoft/deberta-base-mnli",
        max_len=30,
    ):
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.detection_model.to("cuda")
        self.detection_model.eval()

def get_bert_nli_model(model_name: str):
    global _BERT_NLI_SINGLETON
    if _BERT_NLI_SINGLETON is None:
        _BERT_NLI_SINGLETON = BERTNLI(tok_path=model_name, model_path=model_name)
    return _BERT_NLI_SINGLETON
    