import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict

from consistent_agents.agents.base import (
    BaseAgent,
    BaseAgentConfig,
    FormatError,
    Submitted
)
from consistent_agents.models import Model
from consistent_agents.environments import Environment


@dataclass
class AgentConfig(BaseAgentConfig):
    """Configuration for the DefaultAgent."""
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 0
    
@dataclass
class CoTAgentConfig(AgentConfig):
    """Configuration for the DefaultAgent."""
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
    

class DefaultAgent(BaseAgent):
    """Default implementation of an agent that executes bash commands."""
    
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = BaseAgentConfig, **kwargs):
        """Initialize the DefaultAgent"""
        super().__init__(model, env, config_class=config_class, **kwargs)
    
    def parse_action(self, response: Dict[str, Any]) -> Dict[str, Any]:
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
    
    def has_finished(self, output: Dict[str, str]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))