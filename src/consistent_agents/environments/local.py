import subprocess 
import os
from typing import Dict, Any, Optional

from consistent_agents.environments.base import BaseEnvironment

class LocalEnvironment(BaseEnvironment):
    """
    Local environment for executing commands locally.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def start(self) -> bool:
        """Start the local environment."""
        return True
    
    def stop(self) -> bool:
        """Stop the local environment."""
        return True
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the local environment."""
        cwd = cwd or self.config.cwd or os.getcwd()
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            cwd=cwd,
            env=os.environ | self.config.env,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}