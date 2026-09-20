"""Self-contained adapter for the agent-local torch_scaled_demo.py ABI.

No external source tree, private kernel package or sys.path modification.
Installed GPU-enabled PyTorch is required only when build() is called.
Use the same bundled implementation as baseline and candidate for A/A
diagnostics. This adapter does NOT implement MXFP, HTI, or historical kernels.

Implementation ABI:
    run(a, b, scale_a, scale_b, *, out, workspace)
All tensors and workspace are preallocated. Implementations must use PyTorch's
current stream and must not inspect GPU output on the host during timing.
"""
import importlib.util
import sys
from pathlib import Path


def _load(path, name):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"missing implementation file: {path}")
    spec = importlib.util.spec_from_file_location(f"_skill_demo_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        if not callable(getattr(module, "run", None)):
            raise ValueError("implementation must define callable run()")
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def build(baseline_path, candidate_path, shape, slots):
    import torch

    m, n, k = shape
    if min(m, n, k, slots) <= 0:
        raise ValueError("shape and slots must be positive")
    if not torch.cuda.is_available():
        raise ValueError("GPU-enabled PyTorch is required")
    # Per slot: BF16 A/B/output/matmul workspace, FP32 scaled workspace/scales.
    slot_bytes = 2 * (m * k + n * k) + 8 * m * n + 4 * (m + n)
    validation_bytes = 8 * (m * k + n * k + 4 * m * n)
    free, _ = torch.cuda.mem_get_info()
    if slots * slot_bytes + validation_bytes > free * .7:
        raise ValueError("rotary pool + validation estimate exceeds 70% of free memory")

    inputs, outputs, workspaces = [], [], []
    for _ in range(slots):
        # Dyadic small values make a stable arithmetic diagnostic, not a
        # representative quantization/performance workload.
        a = (torch.randint(-2, 3, (m, k), device="cuda").float() / 8).bfloat16()
        b = (torch.randint(-2, 3, (n, k), device="cuda").float() / 8).bfloat16().t()
        sa = torch.ones(m, device="cuda", dtype=torch.float32) * .5
        sb = torch.ones(n, device="cuda", dtype=torch.float32) * .25
        inputs.append((a, b, sa, sb))
        outputs.append(torch.empty((m, n), device="cuda", dtype=torch.bfloat16))
        workspaces.append({
            "matmul": torch.empty_like(outputs[-1]),
            "scaled": torch.empty((m, n), device="cuda", dtype=torch.float32),
        })
    modules = {
        "baseline": _load(baseline_path, "baseline"),
        "candidate": _load(candidate_path, "candidate"),
    }
    runners = {}
    for name, module in modules.items():
        def run(slot, fn=module.run):
            fn(*inputs[slot], out=outputs[slot], workspace=workspaces[slot])
        runners[name] = run

    def validate(selected):
        for slot, (a, b, sa, sb) in enumerate(inputs):
            # Match the documented intermediate BF16 rounding explicitly.
            ref = ((a.float() @ b.float()).bfloat16().float()
                   * sa[:, None] * sb[None, :]).bfloat16()
            baseline = None
            for run in selected.values():
                outputs[slot].fill_(float("nan"))
                run(slot)
                torch.cuda.synchronize()
                torch.testing.assert_close(outputs[slot].float(), ref.float(),
                                           atol=.01, rtol=.01)
                if baseline is None:
                    baseline = outputs[slot].clone()
                else:
                    torch.testing.assert_close(outputs[slot], baseline, atol=0, rtol=0)

    return {
        "runners": runners, "validate": validate,
        "keepalive": (inputs, outputs, workspaces, modules),
        "metadata": {
            "purpose": "agent-local multi-operation PyTorch A/A plumbing diagnostic",
            "not_mxfp_or_hti": True,
            "layout": "nt", "input_dtype": "bfloat16", "output_dtype": "bfloat16",
            "intermediate_matmul_dtype": "bfloat16",
            "scale_dtype": "float32", "bias": False,
            "resident_slot_bytes_estimate": slot_bytes,
            "resident_pool_bytes_estimate": slots * slot_bytes,
            "source_tree_dependencies": [],
        },
    }
