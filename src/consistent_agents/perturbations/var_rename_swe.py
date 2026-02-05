import re
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from consistent_agents.perturbations.base import BasePerturbation

class VariableRenamePerturbation(BasePerturbation):
    """
    Locates source file in base commit, identifies and perturbs a common variablename
    """

    RENAME_PATTERNS = [
        ("result", "res"),
        ("value", "val"),
        ("data", "d"),
        ("item", "elem"),
        ("index", "idx"),
        ("count", "cnt"),
        ("temp", "tmp"),
        ("response", "resp"),
        ("request", "req"),
        ("config", "cfg"),
        ("output", "out"),
        ("input", "inp"),
        ("message", "msg"),
        ("error", "err"),
        ("buffer", "buf"),
        ("length", "_len"),
        ("size", "sz"),
        ("node", "n"),
        ("prev", "previous"),
        ("_next", "nxt"),
        ("curr", "current")
    ]

    SOURCE_EXTENSIONS = [
        "py", "js", "ts", "jsx", "tsx",    # Python. TypeScript/JavaScript
        "java", "kt",                      # JVM
        "c", "cpp", "cc", "h", "hpp",      # C/C++
        "go",                              # Go
        "rs",                              # Rust
        "rb",                              # Ruby
        "php",                             # PHP
        "cs",                              # C#
        "swift",                           # Swift
        "scala",                           # Scala
    ]

    EXCLUDE_DIRS = {
        ".git", "__pycache__", "node_modules", "vendor", 
        ".venv", "venv", "env", ".tox", ".pytest_cache",
        "build", "dist", "egg-info"
    }

    modifies_task_dir = True
    modifies_code = False

    def __init__(
            self, 
            seed : Optional[int] = None,
            name: Optional[str] = None,
            max_files: int = 50,
            **kwargs
    ):
        super().__init__(name= name or 'var_rename',**kwargs)
        self.seed = seed
        self.rng = random.Random(seed if seed is not None else 42)
        self.max_files = max_files
        self.last_result: Optional[Dict[str, Any]] = None

    def _find_source_files(self, env, directory: str = "/testbed") -> list[str]:
        """Find source files excluding tests and hidden directories."""
        source_files = []
        
        for path in directory.rglob("*"):
            # Skip excluded directories
            if any(excl in path.parts for excl in self.EXCLUDE_DIRS):
                continue
            # Skip test files/directories
            if "test" in path.name.lower() and path.is_file():
                continue
            # Check extension
            if path.is_file() and path.suffix.lstrip(".") in self.SOURCE_EXTENSIONS:
                source_files.append(path)
            
            if len(source_files) >= self.max_files:
                break
        
        return source_files
    
    def _find_renameable_variable(self, content: str) -> Optional[tuple[str, str]]:
        """Find a variable that can be renamed."""
        patterns = list(self.RENAME_PATTERNS)
        self.rng.shuffle(patterns)
        
        for old_name, new_name in patterns:
            # Check if old_name exists as a word boundary match
            if re.search(rf'\b{re.escape(old_name)}\b', content):
                # Ensure new_name doesn't already exist
                if not re.search(rf'\b{re.escape(new_name)}\b', content):
                    return (old_name, new_name)
            
            # Try reverse direction
            if re.search(rf'\b{re.escape(new_name)}\b', content):
                if not re.search(rf'\b{re.escape(old_name)}\b', content):
                    return (new_name, old_name)
        
        return None
    
    def _rename_in_file(self, filepath: Path, old_name: str, new_name: str) -> bool:
        """Rename variable in a file using regex word boundaries."""
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            new_content = re.sub(rf'\b{re.escape(old_name)}\b', new_name, content)
            
            if new_content != content:
                filepath.write_text(new_content, encoding="utf-8")
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Failed to rename in {filepath}: {e}")
            return False

    def apply_to_task_dir(
        self,
        task_dir: Path,
        instance_id: str,
        problem_statement: str,
        repo: Optional[str] = None,
        base_commit: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply variable rename perturbation to source files in the task directory.
        """
        task_dir = Path(task_dir)
        env_dir = task_dir / "environment"
        #  Add a setup script that renames variables after git reset
        setup_script = self._create_rename_script(task_dir)
        
        if setup_script:
            # Update Dockerfile to run the rename script
            self._inject_rename_into_dockerfile(env_dir, setup_script)
            
            self.last_result = {
                "success": True,
                "task_dir": str(task_dir),
                "instance_id": instance_id,
                "method": "dockerfile_injection",
                "setup_script": str(setup_script),
                "perturbation": "var_rename",
            }
        else:
            self.last_result = {
                "success": False,
                "task_dir": str(task_dir),
                "instance_id": instance_id,
                "error": "Could not create rename script",
                "perturbation": "var_rename",
            }
        
        return self.last_result

    def _create_rename_script(self, task_dir: Path) -> Optional[Path]:
        """Create a shell script that performs the variable rename at container startup."""
        patterns = list(self.RENAME_PATTERNS)
        self.rng.shuffle(patterns)
        old_name, new_name = patterns[0]
        
        self._pending_rename = (old_name, new_name)
        
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = env_dir / "apply_var_rename.sh"
        
        # Create script that renames variables in Python files
        script_content = f'''#!/bin/bash
        # Variable rename perturbation script
        # Renames: {old_name} -> {new_name}

        set -e

        cd /testbed

        # Find Python files and apply rename (exclude tests and hidden dirs)
        find . -name "*.py" -type f \\
            ! -path "./.git/*" \\
            ! -path "./__pycache__/*" \\
            ! -path "./test*" \\
            ! -path "./tests/*" \\
            ! -path "./.tox/*" \\
            -exec sed -i 's/\\b{old_name}\\b/{new_name}/g' {{}} \\; 2>/dev/null || true

        echo "Variable rename applied: {old_name} -> {new_name}"
        '''
        
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)
        
        return script_path

    def _inject_rename_into_dockerfile(self, env_dir: Path, script_path: Path) -> None:
        """Inject the rename script execution into the Dockerfile."""
        dockerfile_path = env_dir / "Dockerfile"
        
        if not dockerfile_path.exists():
            self.logger.warning(f"Dockerfile not found at {dockerfile_path}")
            return
        
        content = dockerfile_path.read_text(encoding="utf-8")
        
        # Skip if already modified
        if "apply_var_rename.sh" in content:
            return
        
        # Add script copy and execution
        rename_setup = '''
            # === Variable Rename Perturbation ===
            COPY apply_var_rename.sh /tmp/apply_var_rename.sh
            RUN chmod +x /tmp/apply_var_rename.sh && /tmp/apply_var_rename.sh
            '''
        
        # Insert after WORKDIR /testbed
        if "WORKDIR /testbed" in content:
            content = content.replace(
                "WORKDIR /testbed",
                "WORKDIR /testbed\n" + rename_setup
            )
        else:
            # Append at end
            content += "\n" + rename_setup
        
        dockerfile_path.write_text(content, encoding="utf-8")
    
    def transform_patch(self, patch: str) -> str:
        """
        Transform an oracle patch to match renamed variables.
        """
        if not hasattr(self, '_pending_rename') or not self._pending_rename:
            return patch
        
        old_name, new_name = self._pending_rename
        transformed = re.sub(rf'\b{re.escape(old_name)}\b', new_name, patch)
        return transformed
    
    def apply(self, text: str, **kwargs) -> str:
        return text