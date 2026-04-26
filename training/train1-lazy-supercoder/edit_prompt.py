"""Rewrite train/val parquet prompts to request lazy assembly edits.

Run from the repo root:

    uv run --with pyarrow python training/train1-lazy-supercoder/edit_prompt.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

HERE = Path(__file__).resolve().parent
DATA_FILES = (
    HERE / "data/supercoder_train.parquet",
    HERE / "data/supercoder_val.parquet",
)

PROMPT_RE = re.compile(
    r"C Code:\n\n```c\n(?P<c_code>.*?)```\n\n"
    r"Assembly Code:\n\n```assembly\n(?P<unopt_asm>.*?)```",
    re.DOTALL,
)

PROMPT_TEMPLATE = (
    "Given the following C code and assembly code, your task is to generate "
    "highly optimized x86-64 assembly code.\n"

    "C Code:\n\n"
    "```c\n{c_code}\n```\n\n"
    "Assembly Code:\n\n"
    "```assembly\n{unopt_asm}\n```\n\n"

    "Only output the (lazy edit) optimized assembly code. Do not include any other text. "
    "Wrap the assembly code "
    "in ```assembly``` tags.\n\n"
    
    "How to lazy edit:\n"
    "Use \"// ... existing code ...\" to represent unchanged code blocks. "
    "Include just enough surrounding context to locate each edit precisely.\n\n"
    
    "Example format:\n"
    "// ... existing code ...\n"
    "FIRST_EDIT\n"
    "// ... existing code ...\n"
    "SECOND_EDIT\n"
    "// ... existing code ...\n"
    "\n"

    "Rules:\n"
    "- ALWAYS use \"// ... existing code ...\" for unchanged sections (omitting this marker will cause deletions)\n"
    "- Include minimal context around edits only when needed for disambiguation\n"
    "- Preserve exact indentation\n"
    "- For deletions: show context before and after, omit the deleted lines\n"
    "- Batch multiple edits to the same file in one call\n"

    "\nOptimized (lazy edit) Assembly Code:\n"
)


def _extract_code(prompt: list[dict[str, str]]) -> tuple[str, str]:
    if not prompt or prompt[0].get("role") != "user":
        raise ValueError("expected prompt to start with a user message")

    match = PROMPT_RE.search(prompt[0].get("content", ""))
    if match is None:
        raise ValueError("could not extract C/assembly blocks from prompt")

    return match.group("c_code").rstrip("\n"), match.group("unopt_asm").rstrip("\n")


def _strip_assembly_fence(assembly: str) -> str:
    assembly = assembly.strip()
    if assembly.startswith("```assembly"):
        assembly = assembly[len("```assembly") :].lstrip("\n")
    elif assembly.startswith("```asm"):
        assembly = assembly[len("```asm") :].lstrip("\n")
    if assembly.endswith("```"):
        assembly = assembly[: -len("```")].rstrip("\n")
    return assembly


def _rewrite_prompt(row: dict) -> list[dict[str, str]]:
    c_code, parsed_asm = _extract_code(row["prompt"])
    extra_info = row.get("extra_info") or {}
    unopt_asm = _strip_assembly_fence(extra_info.get("unoptimized_assembly") or parsed_asm)

    return [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(c_code=c_code, unopt_asm=unopt_asm),
        }
    ]


def update_file(path: Path, *, dry_run: bool = False) -> int:
    table = pq.read_table(path)
    rows = table.to_pylist()
    prompts = [_rewrite_prompt(row) for row in rows]

    if not dry_run:
        prompt_idx = table.schema.get_field_index("prompt")
        updated = table.set_column(prompt_idx, "prompt", pa.array(prompts, type=table.schema.field("prompt").type))
        pq.write_table(updated, path)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate prompt rewriting without writing parquet files")
    args = parser.parse_args()

    for path in DATA_FILES:
        count = update_file(path, dry_run=args.dry_run)
        action = "validated" if args.dry_run else "updated"
        print(f"{action} {count} prompts in {path}")


if __name__ == "__main__":
    main()


