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

        try:
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
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': '',
                'exit_code': result.returncode,
                'output': result.stdout,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout or self.config.timeout}s: {command}")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout or self.config.timeout}s',
                'exit_code': -1,
                'output': '',
                'returncode': -1
            }
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'output': '',
                'returncode': -1
            }