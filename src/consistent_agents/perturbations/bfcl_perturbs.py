from __future__ import annotations

import json
import re
import random
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import nltk
import logging
from nltk.corpus import wordnet as wn

from consistent_agents.perturbations.base import BasePerturbation

_FUNC_HEADER_RE = re.compile(r"^### (.+)$", re.MULTILINE)

_PARAM_LINE_RE = re.compile(r"^- `.+` \(.+\): .+$", re.MULTILINE)

_logger = logging.getLogger(__name__)

try:
    wn.synsets("test")
except LookupError:
    _logger.info("WordNet corpus not found — downloading automatically …")
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def _wordnet_synonyms(word: str, pos: Optional[str] = None) -> List[str]:
    synsets = wn.synsets(word, pos=pos)
    lemmas: set[str] = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().lower()
            if name != word.lower():
                lemmas.add(name)
    return sorted(lemmas)

def _get_verb_synonyms(word: str, rng: random.Random) -> List[str]:
    wn_syns = _wordnet_synonyms(word, pos=wn.VERB if wn else None)
    wn_syns = [s for s in wn_syns if s.replace("_", "").isalpha()]
    
    return wn_syns


def _get_noun_synonyms(word: str, rng: random.Random) -> List[str]:
    wn_syns = _wordnet_synonyms(word, pos=wn.NOUN if wn else None)
    wn_syns = [s for s in wn_syns if s.replace("_", "").isalpha()]
    
    return wn_syns


def _synonym_for_verb(word: str, rng: random.Random) -> str:
    """Return a random synonym for a verb token."""
    candidates = _get_verb_synonyms(word, rng)
    if not candidates:
        return word
    return rng.choice(candidates)


def _synonym_for_param_token(token: str, rng: random.Random) -> str:
    """Return a random synonym for a single parameter-name token."""
    candidates = _get_noun_synonyms(token, rng)
    if not candidates:
        return token
    return rng.choice(candidates)


def _generate_synonym_name(original_name: str, rng: random.Random) -> str:
    """Generate a synonymous function name by replacing verb / noun tokens.

    Splits on ``_`` and ``.``, replaces tokens using WordNet (with fallback),
    and re-joins with ``_``.  If the generated name is identical to the
    original, a ``_v2`` suffix is appended.
    """
    normalised = original_name.replace(".", "_")
    tokens = normalised.split("_")

    new_tokens: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            # First token is usually the verb
            new_tokens.append(_synonym_for_verb(tok, rng))
        else:
            new_tokens.append(_synonym_for_param_token(tok, rng))

    new_name = "_".join(new_tokens)
    if new_name.lower() == normalised.lower():
        new_name += "_v2"
    return new_name


def _generate_synonym_param_name(original_param: str, rng: random.Random) -> str:
    """Generate a synonymous parameter name by replacing tokens.

    Splits on ``_``, replaces each token via WordNet / fallback table, and
    re-joins.  If the result is identical, an ``_alt`` suffix is appended.
    """
    tokens = original_param.split("_")
    new_tokens = [_synonym_for_param_token(tok, rng) for tok in tokens]
    new_param = "_".join(new_tokens)
    if new_param.lower() == original_param.lower():
        new_param += "_alt"
    return new_param


def _build_decoy_from_function_block(
    block: str,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """Parse a markdown function block and produce a synonym-based decoy.

    The decoy has:
    - A synonymous function name (via WordNet / fallback)
    - The same number of parameters, each with a synonymous name
    - Preserved types and required/optional status
    - A slightly rephrased description
    """
    fname = _extract_function_name(block)
    if not fname:
        return None

    new_name = _generate_synonym_name(fname, rng)

    # Extract description
    desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", block)
    original_desc = desc_match.group(1).strip() if desc_match else "No description"

    # Rephrase description slightly by prepending a synonym-style prefix
    desc_prefixes = [
        "Similar to the original — ",
        "An alternative that ",
        "Performs a related operation: ",
        "Equivalent routine that ",
        "Companion function — ",
    ]
    new_desc = rng.choice(desc_prefixes) + original_desc[0].lower() + original_desc[1:]

    # Extract parameters
    param_lines = _PARAM_LINE_RE.findall(block)
    properties: Dict[str, Dict[str, str]] = {}
    required: List[str] = []

    for pl in param_lines:
        m = re.match(
            r"^- `([^`]+)` \(([^,]+),\s*(Required|Optional)\): (.+)$", pl
        )
        if not m:
            continue
        p_name, p_type, p_req, p_desc = m.group(1), m.group(2), m.group(3), m.group(4)
        new_p_name = _generate_synonym_param_name(p_name, rng)
        properties[new_p_name] = {
            "type": p_type.strip(),
            "description": p_desc.strip(),
        }
        if p_req == "Required":
            required.append(new_p_name)

    return {
        "name": new_name,
        "description": new_desc,
        "parameters": {
            "properties": properties,
            "required": required,
        },
    }


def _split_function_blocks(content: str) -> Tuple[str, List[str]]:
    """Split instruction.md into preamble + list of function-block strings.

    Each function block starts with ``### name`` and runs until the next
    ``### `` or end of string.
    """
    positions = [m.start() for m in _FUNC_HEADER_RE.finditer(content)]
    if not positions:
        return content, []
    preamble = content[: positions[0]]
    blocks: List[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(content)
        blocks.append(content[start:end])
    return preamble, blocks


def _extract_function_name(block: str) -> str:
    """Return the function name from a ``### name`` header."""
    m = _FUNC_HEADER_RE.match(block)
    return m.group(1).strip() if m else ""


def _format_decoy_as_markdown(decoy: Dict[str, Any]) -> str:
    """Format a decoy pool entry into the same markdown the adapter emits."""
    name = decoy["name"]
    desc = decoy.get("description", "No description")
    params = decoy.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    lines = [f"### {name}", "", f"**Description:** {desc}", "", "**Parameters:**"]
    if properties:
        for pname, pinfo in properties.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req = "Required" if pname in required else "Optional"
            lines.append(f"- `{pname}` ({ptype}, {req}): {pdesc}")
    else:
        lines.append("- No parameters")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def _json_calls_to_xml(calls: List[Dict]) -> str:
    """Convert a list of ``{func: {params}}`` dicts to an XML string.

    Format::

        <function_calls>
          <invoke name="func_name">
            <parameter name="p1">value</parameter>
          </invoke>
        </function_calls>
    """
    parts = ['<function_calls>']
    for call in calls:
        if not isinstance(call, dict) or len(call) != 1:
            continue
        func_name = list(call.keys())[0]
        func_params = call[func_name]
        parts.append(f'  <invoke name="{func_name}">')
        if isinstance(func_params, dict):
            for pname, pval in func_params.items():
                # Serialise non-string values as JSON inside the element
                if isinstance(pval, str):
                    val_str = pval
                else:
                    val_str = json.dumps(pval)
                parts.append(f'    <parameter name="{pname}">{val_str}</parameter>')
        parts.append('  </invoke>')
    parts.append('</function_calls>')
    return "\n".join(parts)


VALID_MODES = ("decoy_function", "param_reorder", "xml_format")

class BFCLPerturbations(BasePerturbation):
    """BFCL-specific perturbation with three modes.

    Modes
    -----
    decoy_function
        Inject plausible decoy functions into the instruction's tool list.
    param_reorder
        Shuffle the presentation order of parameters in each function schema.
    xml_format
        Rewrite the task so the agent must output XML instead of JSON.
        Also rewrites ``evaluate.py`` and ``solve.sh`` so the oracle and
        verifier accept the new format.
    """

    modifies_task_dir = True
    modifies_code = False

    def __init__(
        self,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        mode: Optional[str] = None,
        num_decoys: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name or "bfcl_perturbation", **kwargs)
        self.seed = seed
        self.rng = random.Random(seed if seed is not None else 42)
        self.num_decoys = max(1, min(num_decoys, 5))
        self.last_result: Optional[Dict[str, Any]] = None
        self._configured_mode = mode

    def _pick_mode(self) -> str:
        if self._configured_mode is not None:
            return self._configured_mode
        return self.rng.choice(VALID_MODES)
    
    def apply(self, text: str, **kwargs: Any) -> str:  
        return text
    
    def apply_to_task_dir(
        self,
        task_dir: Path,
        instance_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        task_dir = Path(task_dir)
        mode = self._pick_mode()

        if mode == "decoy_function":
            return self._apply_decoy_function(task_dir, instance_id)
        elif mode == "param_reorder":
            return self._apply_param_reorder(task_dir, instance_id)
        elif mode == "xml_format":
            return self._apply_xml_format(task_dir, instance_id)
        else:
            self.last_result = {
                "success": False,
                "error": f"Unknown mode: {mode}",
                "perturbation": self.name,
                "instance_id": instance_id,
            }
            return self.last_result
    
    def _apply_decoy_function(
        self, task_dir: Path, instance_id: str
    ) -> Dict[str, Any]:
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            self.last_result = {
                "success": False,
                "perturbation": "decoy_function",
                "instance_id": instance_id,
                "error": "instruction.md not found",
            }
            return self.last_result

        content = instruction_path.read_text(encoding="utf-8")
        preamble, blocks = _split_function_blocks(content)

        # Collect existing function names (normalised) to avoid collisions
        existing_names: set[str] = set()
        for blk in blocks:
            fname = _extract_function_name(blk)
            existing_names.add(fname.replace(".", "_").lower())

        source_blocks = list(blocks)
        self.rng.shuffle(source_blocks)

        chosen: List[Dict[str, Any]] = []
        generated_names: set[str] = set()

        for blk in source_blocks:
            if len(chosen) >= self.num_decoys:
                break

            decoy = _build_decoy_from_function_block(blk, self.rng)
            if decoy is None:
                continue

            normalised_new = decoy["name"].replace(".", "_").lower()
            if normalised_new in existing_names or normalised_new in generated_names:
                # Retry once with a fresh draw
                decoy = _build_decoy_from_function_block(blk, self.rng)
                if decoy is None:
                    continue
                normalised_new = decoy["name"].replace(".", "_").lower()
                if normalised_new in existing_names or normalised_new in generated_names:
                    continue

            chosen.append(decoy)
            generated_names.add(normalised_new)

        if not chosen:
            self.last_result = {
                "success": False,
                "perturbation": "decoy_function",
                "instance_id": instance_id,
                "error": "No non-colliding decoys available",
            }
            return self.last_result

        # Format decoys as markdown and insert at random positions
        decoy_blocks = [_format_decoy_as_markdown(d) for d in chosen]
        all_blocks = list(blocks)
        injected_positions: List[Dict[str, Any]] = []
        for i, dblk in enumerate(decoy_blocks):
            pos = self.rng.randint(0, len(all_blocks))
            all_blocks.insert(pos, dblk)
            injected_positions.append({
                "name": chosen[i]["name"],
                "derived_from": "wordnet",
                "position": pos,
            })

        # Reassemble
        new_content = preamble + "\n".join(all_blocks)
        instruction_path.write_text(new_content, encoding="utf-8")

        self.last_result = {
            "success": True,
            "perturbation": "decoy_function",
            "instance_id": instance_id,
            "mode": "decoy_function",
            "decoys_injected": injected_positions,
            "original_function_count": len(blocks),
            "total_function_count": len(all_blocks),
            "prevent_instruction_rewrite": True,
        }
        return self.last_result
    
    def _apply_param_reorder(
        self, task_dir: Path, instance_id: str
    ) -> Dict[str, Any]:
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            self.last_result = {
                "success": False,
                "perturbation": "param_reorder",
                "instance_id": instance_id,
                "error": "instruction.md not found",
            }
            return self.last_result

        content = instruction_path.read_text(encoding="utf-8")
        preamble, blocks = _split_function_blocks(content)

        reordered_info: List[Dict[str, Any]] = []
        skipped: List[str] = []
        new_blocks: List[str] = []

        for blk in blocks:
            fname = _extract_function_name(blk)
            param_lines = _PARAM_LINE_RE.findall(blk)

            if len(param_lines) < 2:
                # Nothing to shuffle
                new_blocks.append(blk)
                skipped.append(fname)
                continue

            original_order = []
            for pl in param_lines:
                m = re.match(r"^- `([^`]+)`", pl)
                if m:
                    original_order.append(m.group(1))

            shuffled = list(param_lines)
            self.rng.shuffle(shuffled)

            new_order = []
            for pl in shuffled:
                m = re.match(r"^- `([^`]+)`", pl)
                if m:
                    new_order.append(m.group(1))

            # Replace in the block: swap old param lines for shuffled ones
            new_blk = blk
            for old_line in param_lines:
                new_blk = new_blk.replace(old_line, f"__PLACEHOLDER_{id(old_line)}__", 1)
            for old_line, new_line in zip(param_lines, shuffled):
                new_blk = new_blk.replace(f"__PLACEHOLDER_{id(old_line)}__", new_line, 1)

            new_blocks.append(new_blk)
            reordered_info.append({
                "function": fname,
                "original_order": original_order,
                "new_order": new_order,
            })

        new_content = preamble + "\n".join(new_blocks)
        instruction_path.write_text(new_content, encoding="utf-8")

        self.last_result = {
            "success": True,
            "perturbation": "param_reorder",
            "instance_id": instance_id,
            "mode": "param_reorder",
            "functions_reordered": reordered_info,
            "functions_skipped": skipped,
            "prevent_instruction_rewrite": True,
        }
        return self.last_result
    
    def _apply_xml_format(
        self, task_dir: Path, instance_id: str
    ) -> Dict[str, Any]:
        """Rewrite instruction, evaluate.py, and solve.sh for XML output.

        Changes:
        - instruction.md: replace JSON output instructions with XML spec
        - solution/solve.sh: oracle writes XML instead of JSON
        - tests/evaluate.py: evaluator parses XML from /app/result.xml
        """
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            self.last_result = {
                "success": False,
                "perturbation": "xml_format",
                "instance_id": instance_id,
                "error": "instruction.md not found",
            }
            return self.last_result

        instr = instruction_path.read_text(encoding="utf-8")
        instr = self._rewrite_instruction_for_xml(instr)
        instruction_path.write_text(instr, encoding="utf-8")

        solve_path = task_dir / "solution" / "solve.sh"
        files_transformed: List[str] = []
        if solve_path.is_file():
            self._rewrite_solve_sh_for_xml(solve_path)
            files_transformed.append("solution/solve.sh")

        eval_path = task_dir / "tests" / "evaluate.py"
        if eval_path.is_file():
            self._rewrite_evaluate_py_for_xml(eval_path)
            files_transformed.append("tests/evaluate.py")

        self.last_result = {
            "success": True,
            "perturbation": "xml_format",
            "instance_id": instance_id,
            "mode": "xml_format",
            "files_transformed": files_transformed,
            "prevent_instruction_rewrite": True,
        }
        return self.last_result

    def _rewrite_instruction_for_xml(self, content: str) -> str:
        """Replace JSON output references in the instruction with XML spec."""
        xml_output_spec = textwrap.dedent("""\

        ## Output Format

        Write your answer as **XML** to `/app/result.xml`.

        Use the following format:

        ```xml
        <function_calls>
          <invoke name="function_name">
            <parameter name="param1">value1</parameter>
            <parameter name="param2">value2</parameter>
          </invoke>
        </function_calls>
        ```

        If multiple function calls are required, include multiple `<invoke>` elements
        inside the single `<function_calls>` root.
        If no function should be called, write an empty root: `<function_calls></function_calls>`.
        """)
        output_section_re = re.compile(
            r"## Output Format.*?(?=\n## |\Z)", re.DOTALL
        )
        if output_section_re.search(content):
            content = output_section_re.sub(xml_output_spec.strip(), content)
        else:
            content = content.replace("result.json", "result.xml")
            content = content.replace("/app/result.json", "/app/result.xml")
            content += "\n" + xml_output_spec

        content = content.replace("result.json", "result.xml")

        return content

    def _rewrite_solve_sh_for_xml(self, solve_path: Path) -> None:
        """Rewrite solve.sh to write XML to /app/result.xml instead of JSON."""
        content = solve_path.read_text(encoding="utf-8")
        json_match = re.search(
            r"cat\s*>\s*/app/result\.json\s*<<\s*['\"]?ORACLE_EOF['\"]?\s*\n(.*?)\nORACLE_EOF",
            content,
            re.DOTALL,
        )
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                oracle_calls = json.loads(json_str)
            except json.JSONDecodeError:
                oracle_calls = []

            xml_str = _json_calls_to_xml(oracle_calls)
            new_block = f"cat > /app/result.xml << 'ORACLE_EOF'\n{xml_str}\nORACLE_EOF"
            content = content[: json_match.start()] + new_block + content[json_match.end() :]
        else:
            # Fallback: simple string replacement
            content = content.replace("/app/result.json", "/app/result.xml")
            content = content.replace("result.json", "result.xml")

        solve_path.write_text(content, encoding="utf-8")
        solve_path.chmod(0o755)

    def _rewrite_evaluate_py_for_xml(self, eval_path: Path) -> None:
        """Rewrite evaluate.py to parse XML from /app/result.xml.

        We replace the entire evaluation script with one that:
        1. Reads /app/result.xml
        2. Parses it into the same [{func: {params}}] structure
        3. Compares against the same ground truth (which is baked into the
           script as a Python literal — we keep that unchanged)
        """
        content = eval_path.read_text(encoding="utf-8")

        gt_match = re.search(
            r"ground_truth\s*=\s*(\[.*?\])\s*$",
            content,
            re.MULTILINE | re.DOTALL,
        )
        if not gt_match:
            gt_match = re.search(r"ground_truth\s*=\s*(.+?)(?:\n\n|\n\s*try:)", content, re.DOTALL)

        if gt_match:
            ground_truth_literal = gt_match.group(1).strip()
        else:
            ground_truth_literal = "[]"

        new_script = self._generate_xml_evaluate_script(ground_truth_literal)
        eval_path.write_text(new_script, encoding="utf-8")

    @staticmethod
    def _generate_xml_evaluate_script(ground_truth_literal: str) -> str:
        """Return a complete evaluate.py that reads XML and compares to GT."""
        # NOTE: we inline the full script as a string so it is self-contained
        # inside the Docker container (no external imports beyond stdlib).
        return textwrap.dedent(f'''\
            """
            BFCL evaluation script (XML format perturbation).

            Reads /app/result.xml, parses it into function-call dicts,
            and compares against ground truth.
            """
            import json
            import sys
            import xml.etree.ElementTree as ET
            from pathlib import Path


            def load_result_xml():
                result_path = Path("/app/result.xml")
                if not result_path.exists():
                    raise FileNotFoundError(
                        "result.xml not found. Agent must write output to /app/result.xml"
                    )

                tree = ET.parse(str(result_path))
                root = tree.getroot()

                calls = []
                for invoke_el in root.findall("invoke"):
                    func_name = invoke_el.get("name", "")
                    params = {{}}
                    for param_el in invoke_el.findall("parameter"):
                        pname = param_el.get("name", "")
                        pval_raw = (param_el.text or "").strip()
                        # Try to parse as JSON for non-string values
                        try:
                            pval = json.loads(pval_raw)
                        except (json.JSONDecodeError, ValueError):
                            pval = pval_raw
                        params[pname] = pval
                    calls.append({{func_name: params}})
                return calls


            def normalize_function_name(name: str) -> str:
                return name.replace(".", "_")


            def compare_function_calls(predicted, ground_truth):
                if not isinstance(predicted, list):
                    return False
                if len(predicted) == 0 and len(ground_truth) == 0:
                    return True
                if len(predicted) != len(ground_truth):
                    return False

                for i, pred_call in enumerate(predicted):
                    if not isinstance(pred_call, dict) or len(pred_call) != 1:
                        return False

                    pred_func_name = list(pred_call.keys())[0]
                    pred_params = pred_call[pred_func_name]
                    pred_func_name_norm = normalize_function_name(pred_func_name)

                    if i >= len(ground_truth):
                        return False

                    gt_call = ground_truth[i]
                    if not isinstance(gt_call, dict) or len(gt_call) != 1:
                        return False

                    gt_func_name = list(gt_call.keys())[0]
                    gt_params = gt_call[gt_func_name]
                    gt_func_name_norm = normalize_function_name(gt_func_name)

                    if pred_func_name_norm != gt_func_name_norm:
                        return False

                    if not compare_parameters(pred_params, gt_params):
                        return False

                return True


            def compare_parameters(pred_params, gt_params):
                if not isinstance(pred_params, dict) or not isinstance(gt_params, dict):
                    return False

                for param_name, acceptable_values in gt_params.items():
                    if param_name not in pred_params:
                        if not isinstance(acceptable_values, list):
                            acceptable_values = [acceptable_values]
                        if "" not in acceptable_values and None not in acceptable_values:
                            return False
                        continue

                    pred_value = pred_params[param_name]

                    if not isinstance(acceptable_values, list):
                        acceptable_values = [acceptable_values]

                    if len(acceptable_values) == 0:
                        if pred_value != []:
                            return False
                        continue

                    matched = False
                    for acceptable_value in acceptable_values:
                        if values_equal(pred_value, acceptable_value):
                            matched = True
                            break

                    if not matched:
                        return False

                return True


            def values_equal(v1, v2):
                if v2 == "" or v2 is None:
                    return True
                if v1 == v2:
                    return True
                try:
                    if float(v1) == float(v2):
                        return True
                except (ValueError, TypeError):
                    pass
                if str(v1).lower() == str(v2).lower():
                    return True
                if isinstance(v1, list) and isinstance(v2, list):
                    if len(v1) != len(v2):
                        return False
                    return all(values_equal(a, b) for a, b in zip(v1, v2))
                return False


            def main():
                ground_truth = {ground_truth_literal}

                try:
                    result = load_result_xml()

                    if compare_function_calls(result, ground_truth):
                        print("Test passed: Function call matches ground truth")
                        return 0
                    else:
                        print("Test failed: Function call does not match ground truth")
                        print(f"Predicted: {{result}}")
                        print(f"Expected: {{ground_truth}}")
                        return 1

                except Exception as e:
                    print(f"Test failed with error: {{e}}")
                    import traceback
                    traceback.print_exc()
                    return 1


            if __name__ == "__main__":
                exit_code = main()
                sys.exit(exit_code)
        ''')
