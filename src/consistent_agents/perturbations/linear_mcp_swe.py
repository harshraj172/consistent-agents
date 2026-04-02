"""
Usage:
    - As a perturbation in eval_harbor.py
    - Modifies task_dir before Harbor builds the container
"""

import json
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Optional, Dict, Any

from consistent_agents.perturbations.base import BasePerturbation

try:
    import toml
except ImportError:
    toml = None


class LinearMCPPerturbation(BasePerturbation):
    """
    Perturbation that replaces direct problem statement with MCP-based retrieval.
    
    Instead of reading the issue from instruction.md, the agent must:
    1. See instruction: "Check Issue ID {instance_id} and resolve it"
    2. Use MCP tools to query the fake Linear server
    3. Get the actual problem statement from the MCP response
    4. Fix the bug
    """
    
    modifies_task_dir = True  
    manages_instruction = True
    
    def __init__(
        self,
        seed: Optional[int] = None,
        **kwargs
    ):
        self.name = "linear_mcp"
        self.seed = seed
        self.last_result: Optional[Dict[str, Any]] = None
    
    def apply(self, text: str, **kwargs) -> str:
        """
        Returns a generic MCP instruction 
        """
        instance_id = kwargs.get("instance_id", "UNKNOWN_ISSUE")
        return self._build_mcp_instruction(instance_id)
    
    def apply_to_task_dir(
        self,
        task_dir: Path,
        instance_id: str,
        problem_statement: str,
        repo: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Modify the Harbor task directory to use Linear MCP.
        """
        task_dir = Path(task_dir)
        env_dir = task_dir / "environment"
        
        # Rewrite instruction.md
        instruction_path = task_dir / "instruction.md"
        mcp_instruction = self._build_mcp_instruction(instance_id)
        instruction_path.write_text(mcp_instruction + "\n", encoding="utf-8")
        
        # Create MCP data directory and issues.json
        mcp_data_dir = env_dir / "mcp-data"
        mcp_data_dir.mkdir(parents=True, exist_ok=True)
        
        issues_data = self._build_issues_json(
            instance_id=instance_id,
            problem_statement=problem_statement,
            repo=repo or "unknown"
        )
        issues_path = mcp_data_dir / "issues.json"
        issues_path.write_text(json.dumps(issues_data, indent=2) + "\n", encoding="utf-8")
        
        # Create MCP server directory and script
        mcp_servers_dir = env_dir / "mcp-servers"
        mcp_servers_dir.mkdir(parents=True, exist_ok=True)
        
        server_path = mcp_servers_dir / "linear-server.py"
        server_path.write_text(self._get_linear_server_script(), encoding="utf-8")
        server_path.chmod(0o755)
        
        self._update_task_toml(task_dir)
        
        self._update_dockerfile(env_dir)
        
        self.last_result = {
            "success": True,
            "task_dir": str(task_dir),
            "instance_id": instance_id,
            "issues_path": str(issues_path),
            "server_path": str(server_path),
            "perturbation": "linear_mcp",
            "prevent_instruction_rewrite": True,
            "mcp_instruction": mcp_instruction
        }
        
        return self.last_result
    
    def _build_mcp_instruction(self, instance_id: str) -> str:
        """Build instruction telling agent to use MCP tools."""
        return dedent(f"""
            # Task: Resolve Linear Issue

            Check Issue ID `{instance_id}` and resolve it in the codebase.

            ## Instructions

            1. **Query Linear**: Use the Linear MCP tools to retrieve issue `{instance_id}`
            2. **Read Issue Details**: Review the issue description and requirements
            3. **Implement Fix**: Make the necessary code changes to resolve the issue
            4. **Verify**: Ensure your changes work correctly

            ## Available MCP Tools
            - `get_issue`: Get a Linear issue by ID
            - `list_issues`: List all issues, optionally filtered by project
            - `search_issues`: Search issues by query string
            - `get_issue_comments`: Get all comments for an issue

        """)
    
    def _build_issues_json(
        self,
        instance_id: str,
        problem_statement: str,
        repo: str
    ) -> dict:
        """Build issues.json containing the problem statement."""
        return {
            "issues": {
                instance_id: {
                    "id": instance_id,
                    "title": f"Issue {instance_id}",
                    "project_id": repo,
                    "description": problem_statement.strip(),
                    "labels": ["bug", "needs-fix"],
                    "comments": []
                }
            },
            "projects": {
                repo: {
                    "id": repo,
                    "name": repo
                }
            }
        }
    
    def _update_task_toml(self, task_dir: Path) -> None:
        """Update task.toml with MCP server configuration."""
        toml_path = task_dir / "task.toml"
        
        if toml and toml_path.exists():
            config = toml.loads(toml_path.read_text())
        else:
            config = {
                "version": "1.0",
                "metadata": {},
                "verifier": {"timeout_sec": 3000.0},
                "agent": {"timeout_sec": 3000.0},
            }
        
        # Add MCP server configuration under [environment]
        config.setdefault("environment", {})
        config["environment"]["mcp_servers"] = [
            {
                "name": "linear",
                "transport": "stdio",
                "command": "/app/mcp-servers/linear-server.py",
                "args": ["/app/mcp-data/issues.json"],
            }
        ]
        
        # Update metadata
        config.setdefault("metadata", {})
        config["metadata"]["perturbation"] = "linear_mcp"
        
        if toml:
            toml_path.write_text(toml.dumps(config))
        else:
            # Fallback: write as simple TOML manually
            self._write_toml_fallback(toml_path, config)
    
    def _write_toml_fallback(self, path: Path, config: dict) -> None:
        """Write TOML without the toml library."""
        lines = ['version = "1.0"', ""]
        
        if "metadata" in config:
            lines.append("[metadata]")
            for k, v in config["metadata"].items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")
        
        if "verifier" in config:
            lines.append("[verifier]")
            lines.append(f'timeout_sec = {config["verifier"].get("timeout_sec", 3000.0)}')
            lines.append("")
        
        if "agent" in config:
            lines.append("[agent]")
            lines.append(f'timeout_sec = {config["agent"].get("timeout_sec", 3000.0)}')
            lines.append("")
        
        env_cfg = config.get("environment", {})
        if "mcp_servers" in env_cfg:
            for server in env_cfg["mcp_servers"]:
                lines.append("[[environment.mcp_servers]]")
                lines.append(f'name = "{server["name"]}"')
                lines.append(f'transport = "{server["transport"]}"')
                lines.append(f'command = "{server["command"]}"')
                args_str = ", ".join(f'"{a}"' for a in server.get("args", []))
                lines.append(f"args = [{args_str}]")
                lines.append("")
        
        path.write_text("\n".join(lines), encoding="utf-8")
    
    def _update_dockerfile(self, env_dir: Path) -> None:
        """Update Dockerfile to copy MCP files into container."""
        dockerfile_path = env_dir / "Dockerfile"
        
        if not dockerfile_path.exists():
            return
        
        content = dockerfile_path.read_text()
        
        # Skip if already has MCP setup
        if "/app/mcp-servers" in content:
            return
        
        mcp_setup = """
        # === MCP Server Setup ===
        RUN mkdir -p /app/mcp-servers /app/mcp-data
        COPY mcp-servers/ /app/mcp-servers/
        COPY mcp-data/ /app/mcp-data/
        RUN chmod +x /app/mcp-servers/*.py
        # Install MCP using uv (preferred) or pip as fallback
        RUN if command -v uv &> /dev/null; then \\
                uv pip install --system mcp 2>/dev/null || true; \\
            elif command -v pip &> /dev/null; then \\
                pip install mcp --break-system-packages 2>/dev/null || pip install mcp || true; \\
            fi
        """
        
        # Insert before WORKDIR /testbed if present, otherwise append
        if "WORKDIR /testbed" in content:
            lines = content.split("\n")
            new_lines = []
            inserted = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                if "WORKDIR /testbed" in line and not inserted:
                    new_lines.append("")
                    new_lines.append(mcp_setup.strip())
                    inserted = True
            
            content = "\n".join(new_lines)
        else:
            content += "\n\n" + mcp_setup.strip()
        dockerfile_path.write_text(content, encoding="utf-8")
    
    def _get_linear_server_script(self) -> str:
        """Return the Linear MCP server script."""
        lines = [
            '#!/usr/bin/env python3',
            'import asyncio',
            'import json',
            'import sys',
            'from pathlib import Path',
            '',
            'try:',
            '    from mcp.server import Server',
            '    from mcp.server.stdio import stdio_server',
            '    from mcp.types import Tool, TextContent',
            'except ImportError:',
            '    print("MCP library not available", file=sys.stderr)',
            '    sys.exit(1)',
            '',
            'DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/mcp-data/issues.json")',
            '',
            'try:',
            '    data = json.loads(DATA_PATH.read_text()) if DATA_PATH.exists() else {"issues": {}, "projects": {}}',
            'except Exception as e:',
            '    print(f"Failed to load data: {e}", file=sys.stderr)',
            '    data = {"issues": {}, "projects": {}}',
            '',
            'server = Server("linear")',
            '',
            '',
            '@server.list_tools()',
            'async def list_tools():',
            '    return [',
            '        Tool(',
            '            name="get_issue",',
            '            description="Get a Linear issue by ID",',
            '            inputSchema={',
            '                "type": "object",',
            '                "properties": {"issue_id": {"type": "string", "description": "The issue ID"}},',
            '                "required": ["issue_id"]',
            '            }',
            '        ),',
            '        Tool(',
            '            name="list_issues",',
            '            description="List all issues, optionally filtered by project or status",',
            '            inputSchema={',
            '                "type": "object",',
            '                "properties": {',
            '                    "project_id": {"type": "string"},',
            '                    "status": {"type": "string"}',
            '                }',
            '            }',
            '        ),',
            '        Tool(',
            '            name="search_issues",',
            '            description="Search issues by query string",',
            '            inputSchema={',
            '                "type": "object",',
            '                "properties": {"query": {"type": "string"}},',
            '                "required": ["query"]',
            '            }',
            '        ),',
            '        Tool(',
            '            name="get_issue_comments",',
            '            description="Get all comments for an issue",',
            '            inputSchema={',
            '                "type": "object",',
            '                "properties": {"issue_id": {"type": "string"}},',
            '                "required": ["issue_id"]',
            '            }',
            '        ),',
            '    ]',
            '',
            '',
            '@server.call_tool()',
            'async def call_tool(name: str, arguments: dict):',
            '    if name == "get_issue":',
            '        issue_id = arguments.get("issue_id")',
            '        issue = data.get("issues", {}).get(issue_id)',
            '        result = json.dumps(issue, indent=2) if issue else json.dumps({"error": f"Issue not found: {issue_id}"})',
            '        return [TextContent(type="text", text=result)]',
            '',
            '    elif name == "list_issues":',
            '        issues = list(data.get("issues", {}).values())',
            '        if arguments.get("project_id"):',
            '            issues = [i for i in issues if i.get("project_id") == arguments["project_id"]]',
            '        if arguments.get("status"):',
            '            issues = [i for i in issues if i.get("status") == arguments["status"]]',
            '        return [TextContent(type="text", text=json.dumps(issues, indent=2))]',
            '',
            '    elif name == "search_issues":',
            '        query = arguments.get("query", "").lower()',
            '        results = [',
            '            issue for issue in data.get("issues", {}).values()',
            '            if query in issue.get("title", "").lower()',
            '            or query in issue.get("description", "").lower()',
            '            or query in issue.get("id", "").lower()',
            '        ]',
            '        return [TextContent(type="text", text=json.dumps(results, indent=2))]',
            '',
            '    elif name == "get_issue_comments":',
            '        issue_id = arguments.get("issue_id")',
            '        issue = data.get("issues", {}).get(issue_id)',
            '        result = json.dumps(issue.get("comments", []), indent=2) if issue else json.dumps({"error": f"Issue not found: {issue_id}"})',
            '        return [TextContent(type="text", text=result)]',
            '',
            '    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]',
            '',
            '',
            'async def main():',
            '    async with stdio_server() as (read_stream, write_stream):',
            '        await server.run(read_stream, write_stream, server.create_initialization_options())',
            '',
            '',
            'if __name__ == "__main__":',
            '    asyncio.run(main())',
        ]
        return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    p = LinearMCPPerturbation(seed=42)
    
    print(p.apply("", instance_id="django__django-12345"))
    print()
    
    issues = p._build_issues_json(
        instance_id="django__django-12345",
        problem_statement="Model.save() fails with PostgreSQL when using multi-table inheritance.",
        repo="django/django"
    )
    print(json.dumps(issues, indent=2))