import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from consistent_agents.environments.base import BaseEnvironment


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""


class DockerEnvironment(BaseEnvironment):
    def __init__(
        self,
        name: str = "docker",
        config_class: type = DockerEnvironmentConfig,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        super().__init__(name=name, config={})
        
        if logger:
            self.logger = logger
        
        self.container_id: Optional[str] = None
        self.docker_config = config_class(**kwargs)
        
        self.start()

    def get_template_vars(self) -> Dict[str, Any]:
        """Get configuration as dictionary for templating."""
        return asdict(self.docker_config)

    def start(self) -> bool:
        """Start the Docker container."""
        if self.is_running:
            self.logger.warning(f"Environment {self.name} is already running")
            return True
        
        try:
            container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
            cmd = [
                self.docker_config.executable,
                "run",
                "-d",
                "--name",
                container_name,
                "-w",
                self.docker_config.cwd,
                *self.docker_config.run_args,
                self.docker_config.image,
                "sleep",
                self.docker_config.container_timeout,
            ]
            
            self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.docker_config.pull_timeout,
                check=True,
            )
            
            self.container_id = result.stdout.strip()
            self.is_running = True
            self.logger.info(f"Started container {container_name} with ID {self.container_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start container: {e.stderr}")
            self.is_running = False
            return False
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout while starting container (pull_timeout={self.docker_config.pull_timeout}s)")
            self.is_running = False
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error starting container: {e}")
            self.is_running = False
            return False

    def stop(self) -> bool:
        """Stop and remove the Docker container"""
        if not self.is_running or self.container_id is None:
            self.logger.warning(f"Environment {self.name} is not running")
            return True
        
        try:
            # Try graceful stop first
            stop_cmd = [self.docker_config.executable, "stop", self.container_id]
            result = subprocess.run(
                stop_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Graceful stop failed, forcing removal: {result.stderr}")
                rm_cmd = [self.docker_config.executable, "rm", "-f", self.container_id]
                subprocess.run(rm_cmd, capture_output=True, text=True, timeout=30)
            
            self.is_running = False
            self.container_id = None
            self.logger.info(f"Stopped environment {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping container: {e}")
            self.is_running = False
            self.container_id = None
            return False

    def execute(self, command: str, cwd: str = "", timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Execute a command in the Docker container."""
        if not self.is_running or self.container_id is None:
            self.logger.error(f"Cannot execute command: environment {self.name} is not running")
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Environment not running',
                'exit_code': -1,
                'output': '',
                'returncode': -1
            }
        
        cwd = cwd or self.docker_config.cwd
        
        try:
            cmd = [self.docker_config.executable, "exec", "-w", cwd]
            
            # Add forwarded environment variables
            for key in self.docker_config.forward_env:
                if (value := os.getenv(key)) is not None:
                    cmd.extend(["-e", f"{key}={value}"])
            
            # Add explicit environment variables
            for key, value in self.docker_config.env.items():
                cmd.extend(["-e", f"{key}={value}"])
            
            cmd.extend([self.container_id, "bash", "-lc", command])
            
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.docker_config.timeout,
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
            self.logger.error(f"Command timed out after {timeout or self.docker_config.timeout}s: {command}")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout or self.docker_config.timeout}s',
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

    def is_healthy(self) -> bool:
        """
        Check if the Docker container is healthy and responsive.
        """
        if not self.is_running or self.container_id is None:
            return False
        
        try:
            inspect_cmd = [
                self.docker_config.executable,
                "inspect",
                "-f",
                "{{.State.Running}}",
                self.container_id
            ]
            result = subprocess.run(
                inspect_cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode != 0 or result.stdout.strip().lower() != "true":
                self.logger.warning(f"Container {self.container_id} is not running")
                self.is_running = False
                return False
            
            test_result = self.execute("echo healthy", timeout=5)
            return test_result['success']
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def cleanup(self):
        """Stop and remove the Docker container (non-blocking)."""
        if self.container_id is not None:
            cmd = (
                f"(timeout 60 {self.docker_config.executable} stop {self.container_id} || "
                f"{self.docker_config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            )
            subprocess.Popen(cmd, shell=True)
            self.is_running = False
            self.container_id = None

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Docker environment."""
        status = super().get_status()
        status.update({
            'container_id': self.container_id,
            'image': self.docker_config.image,
            'cwd': self.docker_config.cwd
        })
        return status