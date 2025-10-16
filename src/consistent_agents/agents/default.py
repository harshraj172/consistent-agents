import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from jinja2 import StrictUndefined, Template

from consistent_agents.models import BaseModel
from consistent_agents.environments import BaseEnvironment


# Exceptions
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


# Configuration
@dataclass
class AgentConfig:
    """Configuration for the DefaultAgent."""
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the last line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 10
    cost_limit: float = 2


@dataclass
class CoTAgentConfig(AgentConfig):
    """Configuration for Chain-of-Thought agent."""
    system_template: str = (
        "You are a helpful assistant that can interact with a computer.\n"
        "Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\n"
        "Include a THOUGHT section before your command where you explain your reasoning process.\n"
        "Format your response as shown in <format_example>.\n\n"
        "<format_example>\n"
        "Your reasoning and analysis here. Explain why you want to perform the action.\n"
        "```bash\nyour_command_here\n```\n"
        "</format_example>\n"
        "Failure to follow these rules will cause your response to be rejected."
    )


# Agent
class DefaultAgent:
    """Agent that executes bash commands using an LLM."""
    
    def __init__(
        self, 
        model: BaseModel, 
        env: BaseEnvironment, 
        *, 
        config_class_name: str = "default",
        **kwargs
    ):
        """Initialize the DefaultAgent."""
        if config_class_name == "default":
            config_class = AgentConfig
        elif config_class_name == "cot":
            config_class = CoTAgentConfig
        else:
            raise ValueError(f"Unknown config_class_name: {config_class_name}")
        
        self.config = config_class(**kwargs)
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.env = env
        self.extra_template_vars: Dict[str, Any] = {}
        self.steps: int = 0  

    def render_template(self, template: str, **kwargs) -> str:
        """Render a Jinja2 template with combined context variables."""
        template_vars = asdict(self.config)
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content, **kwargs})
    
    def run(self, task: str, **kwargs) -> Tuple[str, str]:
        """Run the agent until completion."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.steps = 0  
        self.initialize_messages()
        while True:
            try:
                has_reached_step_limit = self.steps >= self.config.step_limit
                if has_reached_step_limit:
                    raise Submitted()
                    
                self.step()
                self.steps += 1

            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
    
    def initialize_messages(self) -> None:
        """Initialize the conversation with system and user prompts."""
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the agent loop."""
        return self.get_observation(self.query())
    
    def query(self) -> Dict[str, Any]:
        """Query the model for a response."""
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response
    
    def get_observation(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and execute the action from the response."""
        action = self.parse_action(response)
        output = self.execute_action(action)
        self.process_observation(output)
        return output

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, str]:
        """Execute a bash command in the environment."""
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
        """Add the observation from command execution to the conversation."""
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
    
    def parse_action(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a bash action from the model's response."""
        actions = re.findall(r"```(?:bash)?\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
    
    def has_finished(self, output: Dict[str, str]) -> None:
        """Check if the agent has completed its task."""
        lines = output.get("output", "").lstrip()
        if lines and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in lines:
            raise Submitted(lines.replace("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ""))