import sys
from unittest.mock import MagicMock


def mock_environment():
    """Mock BaseEnvironment in sys.modules to avoid Python version issues.
    
    Call this before importing any modules that depend on consistent_agents.environments.
    """
    mock_base_environment = MagicMock()
    sys.modules['consistent_agents.environments'] = MagicMock(BaseEnvironment=mock_base_environment)
    sys.modules['consistent_agents.environments.base'] = MagicMock(BaseEnvironment=mock_base_environment)
    sys.modules['consistent_agents.environments.local'] = MagicMock(LocalEnvironment=MagicMock)


class DictLikeItem:
    """A class that supports both dict-style and attribute access."""
    def __init__(self, data):
        self._data = data
        # Set all data as attributes
        for key, value in data.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        """Support dict-style access: item["key"]"""
        return getattr(self, key)
    
    def get(self, key, default=None):
        """Support .get() method: item.get("key", default)"""
        return self._data.get(key, default)


def create_item(data):
    """Create an object that supports both dict and attribute access.
    
    Args:
        data: Dictionary with item data (e.g., {"question": "...", "base_output": "..."})
    
    Returns:
        DictLikeItem object that supports:
        - Attribute access: item.question
        - Dict access: item["question"]
        - .get() method: item.get("question", default)
    """
    return DictLikeItem(data)