#!/usr/bin/env python3
"""Summarize emitted AMD assembly (not llvm-objdump text or a CFG analyzer)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

LABEL = re.compile(r"^\s*([.$A-Za-z_][\w.$]*):(?:\s*;.*)?$")
OP = re.compile(r"^\s*((?:s|v|ds|buffer|tbuffer|global|flat|scratch|image|exp)_[\w.]+|exp)\b(.*)$")
FUNC = re.compile(r"^\s*\.type\s+([^,\s]+)\s*,\s*[@%]function\b")
META = re.compile(r"^\s*(\.[\w.]+)\s*:\s*([0-9]+)\s*$")
META_KEYS = {
    ".vgpr_count", ".sgpr_count", ".agpr_count", ".vgpr_spill_count",
    ".sgpr_spill_count", ".group_segment_fixed_size",
    ".private_segment_fixed_size", ".kernarg_segment_size", ".wavefront_size",
}


def summarize_text(text):
    lines = text.splitlines()
    labels, instructions, functions, metadata = {}, [], [], []
    function = None
    for number, line in enumerate(lines, 1):
        match = FUNC.match(line)
        if match:
            function = match.group(1)
            functions.append({"symbol": function, "declaration_line": number})
        match = LABEL.match(line)
        if match:
            labels[match.group(1)] = (number, function)
        match = META.match(line)
        if match and match.group(1) in META_KEYS:
            # Keep every entry; never silently merge multiple kernel descriptors.
            metadata.append({"line": number, "field": match.group(1), "value": int(match.group(2))})
        match = OP.match(line)
        if match:
            instructions.append({
                "line": number, "function": function,
                "opcode": match.group(1), "operands": match.group(2).strip(),
            })
    loops = []
    for inst in instructions:
        if inst["opcode"] not in ("s_branch",) and not inst["opcode"].startswith("s_cbranch_"):
            continue
        target = inst["operands"].split(",")[0].split(";")[0].strip()
        if target not in labels:
            continue
        start, target_function = labels[target]
        if start >= inst["line"] or target_function != inst["function"]:
            continue
        region = [i for i in instructions if start <= i["line"] <= inst["line"]
                  and i["function"] == inst["function"]]
        loops.append({
            "function": inst["function"], "target": target,
            "start_line": start, "backedge_line": inst["line"],
            "static_instruction_count": len(region),
            "opcode_counts": dict(sorted(Counter(i["opcode"] for i in region).items())),
        })
    return {
        "warning": "Static lexical counts only. Backward branches are loop candidates, "
                   "not verified CFG loops; manually determine trip counts and K-unroll.",
        "functions": functions,
        "metadata_entries": metadata,
        "static_instruction_count": len(instructions),
        "opcode_counts": dict(sorted(Counter(i["opcode"] for i in instructions).items())),
        "backward_branch_regions": loops,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asm", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.asm:
        raw = path.read_bytes()
        result = summarize_text(raw.decode())
        if not result["static_instruction_count"]:
            parser.error(f"{path}: no AMD assembly instructions recognized; use emitted .s text")
        results.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), **result})
    text = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
