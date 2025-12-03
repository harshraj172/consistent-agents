import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
from typing import Optional, Dict, Literal

__all__ = ["HuggingFaceModel", "get_huggingface_model"]

# Model type mapping to AutoModel classes
MODEL_TYPE_MAP = {
    "sequence_classification": AutoModelForSequenceClassification,
    "causal_lm": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "base": AutoModel,
}

# Global cached model instances
_HF_MODEL_CACHE: Dict[str, "HuggingFaceModel"] = {}


class HuggingFaceModel:
    """
    Generic HuggingFace model wrapper that can load different model types.
    
    Parameters:
        model_name (str): HuggingFace model identifier or path.
        model_type (str): Type of model to load. Options:
            - "sequence_classification": AutoModelForSequenceClassification
            - "causal_lm": AutoModelForCausalLM
            - "seq2seq": AutoModelForSeq2SeqLM
            - "base": AutoModel (generic)
        tokenizer_name (Optional[str]): Tokenizer name/path. If None, uses model_name.
        device (str): Device to load model on ("cuda", "cpu", etc.). Default: "cuda".
        **model_kwargs: Additional kwargs passed to from_pretrained.
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: Literal["sequence_classification", "causal_lm", "seq2seq", "base"] = "sequence_classification",
        tokenizer_name: Optional[str] = None,
        device: str = "cuda",
        **model_kwargs
    ):
        if model_type not in MODEL_TYPE_MAP:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Must be one of: {list(MODEL_TYPE_MAP.keys())}"
            )
        
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.tokenizer_name = tokenizer_name or model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        model_class = MODEL_TYPE_MAP[model_type]
        self.model = model_class.from_pretrained(model_name, **model_kwargs)
        
        if device:
            self.model.to(device)
        self.model.eval()

def get_huggingface_model(
    model_name: str,
    model_type: Literal["sequence_classification", "causal_lm", "seq2seq", "base"] = "sequence_classification",
    cache_key: Optional[str] = None,
    **kwargs
) -> HuggingFaceModel:
    """
    Get a HuggingFace model instance, with optional caching.
    
    Parameters:
        model_name: HuggingFace model identifier or path.
        model_type: Type of model to load.
        cache_key: Optional cache key. If None, uses f"{model_name}_{model_type}".
        **kwargs: Additional kwargs passed to HuggingFaceModel.
    
    Returns:
        HuggingFaceModel instance.
    """
    global _HF_MODEL_CACHE
    
    cache_key = cache_key or f"{model_name}_{model_type}"
    
    if cache_key not in _HF_MODEL_CACHE:
        _HF_MODEL_CACHE[cache_key] = HuggingFaceModel(
            model_name=model_name,
            model_type=model_type,
            **kwargs
        )
    
    return _HF_MODEL_CACHE[cache_key]

