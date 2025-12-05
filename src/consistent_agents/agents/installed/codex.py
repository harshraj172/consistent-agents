import os 
import shlex
import json
import tempfile
from pathlib import Path
from abc import abstractmethod

from consistent_agents.agents.installed.base import BaseInstalledAgent, ExecInput
    

class Codex(BaseInstalledAgent):
    """
    An interface for agents that are installed and run in the environment.
    """
        
    @property
    def _install_agent_script_path(self) -> Path:
        """
        Script to the template for installing the agent in the container.
        """
        return Path(__file__).parent / "install-codex.sh.j2"

    def extract_trajectory(self, environment):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            environment.download(container_path="/root/.codex", host_path=tmpdir_path)

            trajectory_files = list(tmpdir_path.glob("**/*.jsonl"))
            lines = trajectory_files[0].read_text().strip().split('\n')
            self.messages = [json.loads(line) for line in lines]
        
    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name:
            raise ValueError("Model name is required")

        model = self.model_name.split("/")[-1]

        env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        }

        return [
            ExecInput(
                command="""
mkdir -p "$HOME/.codex" && 
cat <<EOF >"$HOME/.codex/auth.json"
{
"OPENAI_API_KEY": "$OPENAI_API_KEY"
}
EOF
                """,
                env=env,
            ),
            ExecInput(
                command=(
                    "codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    f"--model {model} "
                    "--json "
                    "-- "  # end of flags
                    f"{escaped_instruction}"
                ),
                env=env,
            ),
        ]
