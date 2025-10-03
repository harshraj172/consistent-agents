import os
import subprocess 
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from consistent_agents.environments.base import BaseEnvironment

REPO_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class LocalEnvironmentConfig:
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    
class LocalEnvironment(BaseEnvironment):
    """
    Local environment for executing commands locally.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = LocalEnvironmentConfig()):
        super().__init__(name, config)
    
    def start(self) -> bool:
        """Start the local environment."""
        return True
    
    def stop(self) -> bool:
        """Stop the local environment."""
        return True
    
    def execute(self, command: str, *, timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the local environment."""
        agent_scratchpad = REPO_ROOT / "agent_scratchpad"
        agent_scratchpad.mkdir(exist_ok=True)

        result = subprocess.run(
            command,
            shell=True,
            text=True,
            cwd=agent_scratchpad,
            env=os.environ | self.config.env,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}