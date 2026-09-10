"""Adapter for current repository scaled_gemm: NT/HTI/BF16/no-bias only.

Adapter contract:
  build(baseline_path, candidate_path, shape, slots) -> dict
  runners: {baseline: callable(slot), candidate: callable(slot)}
  validate(runners): independent reference checks, outside measurement
  metadata: JSON-serializable policy/dependency/memory information
  keepalive: retain all input/output/auxiliary storage for Graph lifetime

A runner uses torch.cuda.current_stream, has no host output inspection, and
updates a preallocated output. If dependencies change, use isolated snapshots
rather than this same-process adapter.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys


def _load(path, name):
    spec = importlib.util.spec_from_file_location(f"kernels.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(baseline_path, candidate_path, shape, slots):
    import torch
    # Locate dependencies via the candidate, so the skill can live outside
    # this checkout (e.g. an agent's installed skills directory).
    repo = candidate_path.resolve().parent.parent
    if candidate_path.parent.name != 'kernels' or not (repo / 'kernels' / 'common.py').is_file():
        raise ValueError('candidate must be /path/to/repo/kernels/scaled_gemm_gfx950.py')
    sys.path.insert(0, str(repo))
    m, n, k = shape
    if m <= 0 or n % 8 or k < 256 or k % 256:
        raise ValueError("NT HTI adapter requires M>0, N divisible by8, K>=256 divisible by256")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if getattr(props, "gcnArchName", "").split(":")[0] != "gfx950":
        raise ValueError("this example adapter targets gfx950 only")
    slot_bytes = m * k + n * k + 2 * m * n + 4 * (m + n)
    free, _ = torch.cuda.mem_get_info()
    # Conservative allowance for validation FP32 casts, matmul result and cloning.
    validation_bytes = 8 * (m * k + n * k + 4 * m * n)
    if slots * slot_bytes + validation_bytes > free * .7:
        raise ValueError("requested rotary pool + validation estimate exceeds 70% of free GPU memory")
    kwargs = dict(
        block_m=256, block_n=256, block_k=128, stages=2,
        m_waves=2, n_waves=4, k_waves=1, split_k=1, group_m=0,
        use_half_tile_interleaved=True,
    )
    inputs, outputs = [], []
    for _ in range(slots):
        a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1).to(torch.float8_e4m3fn)
        b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1).to(torch.float8_e4m3fn).t()
        sa = torch.rand(m, device="cuda") * .25 + .05
        sb = torch.rand(n, device="cuda") * .25 + .05
        inputs.append((a, b, sa, sb))
        outputs.append(torch.empty((m, n), device="cuda", dtype=torch.bfloat16))
    modules = {
        "baseline": _load(baseline_path, "_skill_gemm_baseline"),
        "candidate": _load(candidate_path, "_skill_gemm_candidate"),
    }
    runners = {}
    for name, module in modules.items():
        def run(slot, fn=module.scaled_gemm):
            fn(*inputs[slot], out=outputs[slot], user_kwargs=kwargs, layout="nt")
        runners[name] = run

    def validate(selected):
        a, b, sa, sb = inputs[0]
        ref = ((a.float() @ b.float()) * sa[:, None] * sb[None, :]).bfloat16()
        baseline = None
        for name, run in selected.items():
            outputs[0].fill_(float("nan"))
            run(0)
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[0].float(), ref.float(), atol=.05, rtol=.02)
            if baseline is None:
                baseline = outputs[0].clone()
            else:
                torch.testing.assert_close(outputs[0], baseline, atol=0, rtol=0)
        # The reference tolerance is for this adapter's finite random input
        # distribution only. Full-path exact/edge tests remain a separate gate.

    dependencies = {}
    for filename in ("common.py", "gemm_a16w16_gfx950.py", "gemm_a16w16_gfx950_utils.py"):
        p = repo / "kernels" / filename
        dependencies[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        "runners": runners, "validate": validate,
        "keepalive": (inputs, outputs, modules),
        "metadata": {
            "policy": kwargs, "layout": "nt", "input_dtype": "float8_e4m3fn",
            "output_dtype": "bfloat16", "bias": False,
            "resident_slot_bytes_estimate": slot_bytes,
            "resident_pool_bytes_estimate": slots * slot_bytes,
            "shared_dependencies_sha256": dependencies,
        },
    }
