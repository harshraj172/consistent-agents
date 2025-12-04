import os 
import shlex
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

    def extract_trajectory(self, env):
        trajectory_path = Path(f"~/.codex/sessions")
        trajectory_path = list(trajectory_path.glob("**/*.jsonl"))[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / trajectory_path.name
            env.download(container_path=trajectory_path,
                        host_path=tmp_path)
            with open(tmp_path, "r") as f:
                self.messages = f.read()
        
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
mkdir -p "$HOME/.codex"
cat <<EOF >"$HOME/.codex/auth.json"
{
  "OPENAI_API_KEY": "${OPENAI_API_KEY}"
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
                    f"{escaped_instruction} "
                    "&& rm -rf $CODEX_HOME/auth.json"
                ),
                env=env,
            ),
        ]