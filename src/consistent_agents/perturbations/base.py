from abc import ABC, abstractmethod
from typing import Any, Dict
import logging


class BasePerturbation(ABC):
    """Base class for all perturbation types."""
    
    def __init__(self, name: str = "", **kwargs):
        """
        Initialize the perturbation.
        
        Args:
            name: Name identifier for the perturbation
            **kwargs: Additional configuration parameters
        """
        self.name = name or self.__class__.__name__
        self.config = kwargs
        self.logger = logging.getLogger(f"perturbation.{self.name}")
    
    def apply(self, text: str, **kwargs) -> str:
        """
        Apply the perturbation to the input text.
        
        Args:
            text: Input text to perturb
            **kwargs: Additional parameters for the perturbation
            
        Returns:
            str: Perturbed text with same semantics but different syntax
        """
        return text
    
    def __call__(self, text: str, **kwargs) -> str:
        """Allow perturbation to be called as a function."""
        return self.apply(text, **kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """Get perturbation configuration."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            **self.config
        }