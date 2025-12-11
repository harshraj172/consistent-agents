import re
import random
from typing import Optional, List
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
        ("length", "len"),
        ("size", "sz"),
        ("node", "n"),
        ("prev", "previous"),
        ("next", "nxt"),
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

    modifies_code = True

    def __init__(
            self, 
            seed : int | None = None,
            name: str | None = None,
            base_commit : Optional[str] = None,
            extensions : Optional[List[str]] | None = None,
            **kwargs
    ):
        super().__init__(name= name or 'var_rename',**kwargs)
        self.seed = seed
        if self.seed is not None:
            self.rng = random.Random(seed)
        else: self.rng = random.Random(42)
        
        self.base_commit = base_commit
        self.extensions = extensions if extensions is not None else self.SOURCE_EXTENSIONS
        
        self.result = None

    def _find_source_files(self, env, directory: str = "/testbed") -> list[str]:
        """Find source files excluding tests and hidden directories."""
        ext_pattern = " -o ".join(f'-name "*.{ext}"' for ext in self.extensions)
        cmd = f'find {directory} \\( {ext_pattern} \\) -type f ! -path "*/test*" ! -path "*/.git/*" ! -path "*/node_modules/*" ! -path "*/vendor/*" | head -50'
        result = env.execute(cmd)
        stdout = result.get("stdout", "")
        return [f.strip() for f in stdout.strip().split("\n") if f.strip()]
        
    def _find_renameable_variable(self, env, filepath: str) -> Optional[tuple[str, str]]:
        """Find a variable that can be renamed."""
        result = env.execute(f"cat {filepath}")
        content = result.get("stdout", "")
        
        patterns = list(self.RENAME_PATTERNS)
        self.rng.shuffle(patterns)
        
        for old_name, new_name in patterns:
            if re.search(rf'\b{old_name}\b', content):
                if not re.search(rf'\b{new_name}\b', content):
                    return (old_name, new_name)
            # Try reverse
            if re.search(rf'\b{new_name}\b', content):
                if not re.search(rf'\b{old_name}\b', content):
                    return (new_name, old_name)
        
        return None
    
    def _rename_variable(self, env, filepath: str, old_name: str, new_name: str) -> bool:
        """Rename variable using sed with word boundaries."""
        sed_cmd = f"sed -i 's/\\b{old_name}\\b/{new_name}/g' {filepath}"
        result = env.execute(sed_cmd)
        return result.get("returncode", -1) == 0
    
    def _apply_to_env(self, env, base_commit: Optional[str] = None) -> dict:
        """Apply variable rename perturbation to code in the container."""
        commit = base_commit or self.base_commit
        if commit:
            env.execute(f"cd /testbed && git reset --hard {commit}")
        
        files = self._find_source_files(env)
        self.rng.shuffle(files)
        
        for filepath in files:
            rename_pair = self._find_renameable_variable(env, filepath)
            if rename_pair:
                old_name, new_name = rename_pair
                if self._rename_variable(env, filepath, old_name, new_name):
                    self.last_result = {
                        "success": True,
                        "file": filepath,
                        "old_name": old_name,
                        "new_name": new_name
                    }
                    return self.last_result
        
        self.last_result = {"success": False, "file": None, "old_name": None, "new_name": None}
        return self.last_result
    
    def apply(self, text: str, **kwargs) -> str:
        """Prompt unchanged - perturbation is in the code."""
        return text
