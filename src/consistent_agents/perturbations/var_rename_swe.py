import re
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set 
from consistent_agents.perturbations.base import BasePerturbation

class VariableRenamePerturbation(BasePerturbation):
    """
    Locates source file in base commit, identifies and perturbs a common variable name
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
    modifies_code = True

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
        self._pending_rename: Optional[Tuple[str, str]] = None

    def _find_source_files(self,  directory: Path, include_tests: bool = True) -> list[Path]:
        """Find source files excluding tests and hidden directories."""
        source_files: List[Path] = []
        ext: set[str] = set()
        directory = Path(directory)
        
        for path in directory.rglob("*"):
            if any(excl in path.parts for excl in self.EXCLUDE_DIRS):
                continue
            if not include_tests and "test" in path.name.lower() and path.is_file():
                continue
            if path.is_file() and path.suffix.lstrip(".") in self.SOURCE_EXTENSIONS:
                source_files.append(path)
                ext.add(path.suffix.lstrip("."))
            
            if len(source_files) >= self.max_files:
                break
        
        return source_files, sorted(ext)
    
    
    def _build_find_name_clause(self, extensions: List[str]) -> str:
        """
        Build a ``find`` ``-name`` clause that matches all given extensions.
        """
        if not extensions:
            # Fallback: match every file in SOURCE_EXTENSIONS
            extensions = list(self.SOURCE_EXTENSIONS)

        parts = [f'-name "*.{ext}"' for ext in extensions]
        return "\\( " + " -o ".join(parts) + " \\)"

    def _create_rename_script(self, task_dir: Path, fetched_extensions: Set[str]) -> Optional[Path]:
        """Create a shell script that performs the variable rename at container startup."""
        patterns = list(self.RENAME_PATTERNS)
        self.rng.shuffle(patterns)
        old_name, new_name = patterns[0]
        
        self._pending_rename = (old_name, new_name)
        
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = env_dir / "apply_var_rename.sh"
        
        name_clause = self._build_find_name_clause(fetched_extensions)
        exclude_clauses = " ".join(
            f'! -path "./{d}/*"' for d in sorted(self.EXCLUDE_DIRS)
        )
        # Create script that renames variables in Python files
        script_content = f"""#!/bin/bash
        # Variable rename perturbation script

        set -e
        cd /testbed

        # Find source files (all detected extensions) and apply rename.
        find . {name_clause} -type f \\
            {exclude_clauses} \\
            -exec perl -pi -e 's/\\b{re.escape(old_name)}\\b/{new_name}/g' {{}} + 2>/dev/null || true

        echo "Variable rename applied: {old_name} -> {new_name}"
        """
        
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

        _, fetched_extensions = self._find_source_files(env_dir)
        setup_script = self._create_rename_script(task_dir, fetched_extensions)
        
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
                "rename": list(self._pending_rename) if self._pending_rename else None,
                "fetched_extensions": fetched_extensions
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
    
    _SOLUTION_GLOBS = [
        "*.sh",     
        "*.patch",   
        "*.diff",    
        "*.py",      
        "*.txt",     
    ]

    def _transform_solution_dir(self, task_dir: Path) -> List[str]:
        if not self._pending_rename:
            return []
         
        old_name, new_name = self._pending_rename
        solution_dir = task_dir / "solution"

        if not solution_dir.is_dir():
            return []

        transformed: List[str] = []

        for glob in self._SOLUTION_GLOBS:
            for fpath in solution_dir.rglob(glob):
                if not fpath.is_file():
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    new_content = re.sub(
                        rf"\b{re.escape(old_name)}\b", new_name, content
                    )
                    if new_content != content:
                        fpath.write_text(new_content, encoding="utf-8")
                        transformed.append(str(fpath.relative_to(task_dir)))
                except Exception as e:
                    self.logger.warning(
                        f"Failed to transform solution file {fpath}: {e}"
                    )

        return transformed
    
    def transform_patch(self, patch: str) -> str:
        """
        Transform an oracle patch to match renamed variables.
        """
        if not self._pending_rename:
            return patch
        
        old_name, new_name = self._pending_rename
        transformed = re.sub(rf'\b{re.escape(old_name)}\b', new_name, patch)
        return transformed
    
    def apply(self, text: str, **kwargs) -> str:
        return text