#!/usr/bin/env python3
"""Fair-order GEMM A/B benchmark. Adapter defines call/validation/allocation semantics."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time


def execution_order(names, round_index):
    """Rotate once per pair of rounds; mirror odd rounds. For N=2: AB, BA, BA, AB."""
    if not names:
        raise ValueError("at least one runner required")
    shift = (round_index // 2) % len(names)
    order = list(names[shift:]) + list(names[:shift])
    return order if round_index % 2 == 0 else list(reversed(order))


def sample_stats(values):
    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("latency samples must be finite and positive")
    ordered = sorted(values)
    def percentile(p):
        index = p * (len(ordered) - 1)
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])
    median = statistics.median(values)
    return {
        "median_us": median, "p10_us": percentile(.1), "p90_us": percentile(.9),
        "mad_us": statistics.median(abs(x - median) for x in values),
        "min_us": min(values), "max_us": max(values),
    }


def load_adapter(path):
    name = "_gemm_skill_adapter"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--shape", nargs=3, type=int, default=[8192, 8192, 8192], metavar=("M", "N", "K"))
    parser.add_argument("--slots", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--launches", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8, help="batches/replays per runner")
    parser.add_argument("--mode", choices=("graph", "events", "wall"), default="graph")
    parser.add_argument("--only", choices=("both", "baseline", "candidate"), default="both",
                        help="Compile/profile one variant per process when independent dumps are needed.")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(*args.shape, args.slots, args.rounds, args.launches, args.warmup) <= 0:
        parser.error("shape/slots/rounds/launches/warmup must all be positive")
    import torch
    if not torch.cuda.is_available():
        parser.error("a GPU-enabled PyTorch environment is required")
    for path in (args.adapter, args.baseline, args.candidate):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    source_paths = {'adapter': args.adapter, 'baseline': args.baseline, 'candidate': args.candidate}
    sources = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in source_paths.items()}
    torch.manual_seed(args.seed)
    bundle = load_adapter(args.adapter.resolve()).build(
        args.baseline.resolve(), args.candidate.resolve(), tuple(args.shape), args.slots)
    runners = bundle["runners"]
    if set(runners) != {"baseline", "candidate"}:
        parser.error("adapter runners must be named baseline and candidate")
    if args.only != "both":
        runners = {args.only: runners[args.only]}
    names = list(runners)
    # Adapter must retain tensors (bundle.keepalive) and use the current stream.
    for run in runners.values():
        for slot in range(args.slots):
            run(slot)
    torch.cuda.synchronize()
    bundle["validate"](runners)
    torch.cuda.synchronize()
    batches = {}
    graphs = {}
    for name, run in runners.items():
        if args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(args.launches):
                    run(i % args.slots)
            graphs[name] = graph
            batches[name] = graph.replay
        else:
            def batch(run=run):
                for i in range(args.launches):
                    run(i % args.slots)
            batches[name] = batch
    for r in range(args.warmup):
        for name in execution_order(names, r):
            batches[name]()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {name: [] for name in names}
    orders = []
    for r in range(args.rounds):
        order = execution_order(names, r)
        orders.append(order)
        for name in order:
            if args.mode == "wall":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                batches[name]()
                torch.cuda.synchronize()
                us = (time.perf_counter() - t0) * 1e6 / args.launches
            else:
                start.record()
                batches[name]()
                end.record()
                end.synchronize()
                us = start.elapsed_time(end) * 1000 / args.launches
            samples[name].append(us)
    stats = {name: sample_stats(values) for name, values in samples.items()}
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    import flydsl
    output = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "shape": args.shape, "slots": args.slots, "launches": args.launches,
        "rounds": args.rounds, "warmup_batches": args.warmup, "seed": args.seed,
        "mode": args.mode, "execution_order": orders, "raw_us": samples, "stats": stats,
        "environment": {
            "python": sys.version, "torch": torch.__version__, "HIP": torch.version.hip,
            "flydsl_path": flydsl.__file__, "device": str(props),
            "note": "No clock changes made. Record clocks/power/concurrency externally.",
        },
        "sources": sources,
        "adapter_metadata": bundle.get("metadata", {}),
        "warning": "Same-process imports may share dependencies. Graph is not host API latency; "
                   "profiler runs are not final benchmarks. Validate against an independent oracle.",
    }
    if args.only == "both":
        base, cand = stats["baseline"]["median_us"], stats["candidate"]["median_us"]
        output["latency_change_pct"] = (cand / base - 1) * 100
        output["paired_round_change_pct"] = [(c / b - 1) * 100 for b, c in zip(samples["baseline"], samples["candidate"])]
    for name, path in source_paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources[name]['sha256']:
            raise RuntimeError(f'{name} source changed during measurement; discard this run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"stats": stats, "latency_change_pct": output.get("latency_change_pct"),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
