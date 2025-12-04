from pathlib import Path
from abc import ABC, abstractmethod

from pydantic import BaseModel as PydanticBaseModel

from consistent_agents.models import BaseModel
from consistent_agents.environments import BaseEnvironment


class ExecInput(PydanticBaseModel):
    command: str
    cwd: str | None = "/testbed"
    env: dict[str, str] | None = None
    timeout_sec: int | None = None


class BaseInstalledAgent(ABC):
    """
    An interface for agents that are installed and run in the environment.
    """

    def __init__(
        self,
        model: BaseModel,  
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_name = model.config.model_name
        self.messages = None

    @property
    def _install_agent_script_path(self) -> Path:
        """
        Script to the template for installing the agent in the container.
        """
        pass

    @abstractmethod
    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """
        Create the commands to run the agent in the container. Usually this is a single
        command that passes the instruction to the agent and executes it in headless
        mode.
        """
        pass

    def extract_trajectory(self, environment: BaseEnvironment) -> None:
        pass
        
        
    def setup(self, environment: BaseEnvironment) -> None:
        environment.execute(command="mkdir -p /installed-agent")

        environment.upload(
            host_path=self._install_agent_script_path,
            container_path="/installed-agent/install.sh",
        )

        result = environment.execute(command="bash /installed-agent/install.sh")
        return result

    def run(self, task: str, environment: BaseEnvironment, **kwargs):
        result = self.setup(environment)
        assert result["success"] == True, "Failed to set up the installed agent."
        for exec_input in self.create_run_agent_commands(task):
            result = environment.execute(
                command=exec_input.command,
                cwd=exec_input.cwd,
                env=exec_input.env,
                timeout_sec=exec_input.timeout_sec,
            )
            assert result["success"] == True, f"Failed to run the command: {exec_input.command}."
        self.extract_trajectory(environment)