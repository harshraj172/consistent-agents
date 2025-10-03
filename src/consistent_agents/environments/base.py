from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class BaseEnvironment(ABC):
    """
    Base environment class that defines the common interface for all environment types.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the base environment."""
        self.name = name
        self.config = config or {}
        self.is_running = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}({name})")
    
    @abstractmethod
    def start(self) -> bool:
        """Start the environment."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the environment."""
        pass
    
    def cleanup(self) -> bool:
        """Clean up the environment."""
        pass
    
    @abstractmethod
    def execute(self, command: str, **kwargs) -> Dict[str, Any]:
        """Execute a command in the environment."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the environment is healthy and responsive."""
        pass
    
    def restart(self) -> bool:
        """Restart the environment by stopping and starting it."""
        self.logger.info(f"Restarting environment {self.name}")
        if self.is_running:
            if not self.stop():
                self.logger.error(f"Failed to stop environment {self.name} during restart")
                return False
        
        return self.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the environment."""
        return {
            'name': self.name,
            'is_running': self.is_running,
            'is_healthy': self.is_healthy() if self.is_running else False,
            'config': self.config
        }