import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from jinja2 import StrictUndefined, Template

from consistent_agent.models import Model
from consistent_agent.environments import Environment


# Keep the existing exceptions as they are
class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


@dataclass
class BaseAgentConfig:
    """Base configuration for agents."""
    step_limit: int = 0  
    cost_limit: float = 0 


class BaseAgent(ABC):
    """Abstract base class for agents that interact with models and environments."""
    
    def __init__(
        self, 
        model: Model, 
        env: Environment, 
        *, 
        config_class: Callable = BaseAgentConfig,
        **kwargs
    ):
        """Initialize the base agent"""
        self.config = config_class(**kwargs)
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.env = env
        self.extra_template_vars: Dict[str, Any] = {}
    
    def render_template(self, template: str, **kwargs) -> str:
        """Render a Jinja2 template with combined context variables"""
        template_vars = (
            asdict(self.config) | 
            self.env.get_template_vars() | 
            self.model.get_template_vars()
        )
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content, **kwargs})
    
    def run(self, task: str, **kwargs) -> Tuple[str, str]:
        """Run the agent until completion"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.initialize_messages()
        
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.handle_non_terminating_exception(e)
            except TerminatingException as e:
                return self.handle_terminating_exception(e)
    
    def initialize_messages(self) -> None:
        """Initialize the conversation with system and user prompts."""
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the agent loop"""
        return self.get_observation(self.query())
    
    def query(self) -> Dict[str, Any]:
        """Query the model for a response"""
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response
    
    def get_observation(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and execute the action from the response"""
        action = self.parse_action(response)
        output = self.execute_action(action)
        self.process_observation(output)
        return output

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, str]:
        """Execute a bash command in the environment"""
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output
    
    def process_observation(self, output: Dict[str, str]) -> None:
        """Add the observation from command execution to the conversation"""
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        
    @abstractmethod
    def parse_action(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse an action from the model's response"""
        pass
        
    @abstractmethod
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action in the environment"""
        pass
    
    @abstractmethod
    def process_observation(self, output: Dict[str, Any]) -> None:
        """Process the observation and add it to the conversation"""
        pass
    
    @abstractmethod
    def has_finished(self, output: Dict[str, str]) -> None:
        """Check if the agent has completed its task"""
        pass