"""CPU-only regression tests for skill tools and documentation structure."""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"skill_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asm = load_script("asm_summary")
pmc = load_script("counter_summary")
bench = load_script("benchmark_ab")


class TestBenchmark(unittest.TestCase):
    def test_two_runners_actually_alternate(self):
        orders = [bench.execution_order(["A", "B"], r) for r in range(4)]
        self.assertEqual(orders, [["A", "B"], ["B", "A"], ["B", "A"], ["A", "B"]])

    def test_every_runner_gets_every_position(self):
        for n in range(1, 6):
            names = [str(i) for i in range(n)]
            positions = [Counter() for _ in names]
            for r in range(2 * n):
                for pos, name in enumerate(bench.execution_order(names, r)):
                    positions[pos][name] += 1
            for counts in positions:
                self.assertEqual(counts, Counter({name: 2 for name in names}))

    def test_stats_and_outlier_retained(self):
        result = bench.sample_stats([1, 2, 3, 4, 100])
        self.assertEqual(result["median_us"], 3)
        self.assertEqual(result["mad_us"], 1)
        self.assertEqual(result["max_us"], 100)
        self.assertAlmostEqual(result["p10_us"], 1.4)
        self.assertAlmostEqual(result["p90_us"], 61.6)

    def test_bad_samples_fail(self):
        for values in ([], [0], [-1], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                bench.sample_stats(values)
        with self.assertRaises(ValueError):
            bench.execution_order([], 0)


class TestAsm(unittest.TestCase):
    def test_regions_and_metadata(self):
        text = """
    .type foo,@function
foo:
    s_mov_b32 s0, 0
.LBB0_1:
    ds_read_b128 v[0:3], v4
    v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[0:7], v[12:19], v[8:11]
    s_add_i32 s0, s0, 1
    s_cbranch_scc1 .LBB0_1
    s_endpgm 0
    .vgpr_count: 248
    .private_segment_fixed_size: 0
"""
        result = asm.summarize_text(text)
        self.assertEqual(result["static_instruction_count"], 6)
        self.assertEqual(result["opcode_counts"]["ds_read_b128"], 1)
        self.assertEqual(result["metadata_entries"][0]["value"], 248)
        self.assertEqual(len(result["backward_branch_regions"]), 1)
        loop = result["backward_branch_regions"][0]
        self.assertEqual(loop["static_instruction_count"], 4)
        self.assertEqual(loop["function"], "foo")

    def test_forward_branch_is_not_a_loop(self):
        result = asm.summarize_text(".type f,@function\nf:\n s_branch .L1\n.L1:\n s_endpgm 0")
        self.assertEqual(result["backward_branch_regions"], [])

    def test_no_cross_function_region(self):
        text = ".type f,@function\nf:\n.L1:\n s_endpgm 0\n.type g,@function\ng:\n s_branch .L1\n"
        self.assertEqual(asm.summarize_text(text)["backward_branch_regions"], [])

    def test_keep_multiple_metadata_entries(self):
        result = asm.summarize_text(".vgpr_count: 100\n.vgpr_count: 200\n")
        self.assertEqual([r["value"] for r in result["metadata_entries"]], [100, 200])


class TestCounters(unittest.TestCase):
    @staticmethod
    def row(dispatch="1", value="10", kernel="gemm", agent="Agent 2"):
        return dict(Dispatch_Id=dispatch, Agent_Id=agent, Kernel_Name=kernel,
                    Grid_Size="1024", Workgroup_Size="512",
                    Counter_Name="SQ_LDS_BANK_CONFLICT", Counter_Value=value)

    def test_mean_per_dispatch(self):
        groups = pmc.summarize_rows([self.row(), self.row("2", "30")])
        metric = groups[0]["counters"]["SQ_LDS_BANK_CONFLICT"]
        self.assertEqual(metric["dispatch_count"], 2)
        self.assertEqual(metric["mean_per_dispatch"], 20)

    def test_duplicate_dimensions_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple rows"):
            pmc.summarize_rows([self.row(), self.row(value="20")])
        groups = pmc.summarize_rows([self.row(), self.row(value="20")], sum_dimensions=True)
        self.assertEqual(groups[0]["counters"]["SQ_LDS_BANK_CONFLICT"]["mean_per_dispatch"], 30)

    def test_configs_not_mixed(self):
        groups = pmc.summarize_rows([self.row(), self.row(kernel="other"), self.row(agent="Agent 3")])
        self.assertEqual(len(groups), 3)

    def test_bad_schema_and_nan(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            pmc.summarize_rows([{"Counter_Name": "a"}])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pmc.summarize_rows([self.row(value="nan")])


class TestDocs(unittest.TestCase):
    def test_skill_frontmatter(self):
        text = (ROOT / "SKILL.md").read_text()
        self.assertTrue(text.startswith("---\nname: gpu-gemm-optimization\n"))
        self.assertIn("\ndescription:", text.split("---", 2)[1])

    def test_local_links_exist_and_fences_paired(self):
        for path in ROOT.rglob("*.md"):
            text = path.read_text()
            for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if "://" not in link and not link.startswith("#"):
                    self.assertTrue((path.parent / link.split("#")[0]).exists(), (path, link))
            inside = False
            marker = None
            for line in text.splitlines():
                if line.startswith(("~~~", "```")):
                    current = line[:3]
                    if inside:
                        self.assertEqual(current, marker, path)
                    else:
                        marker = current
                    inside = not inside
            self.assertFalse(inside, f"Unclosed code fence in {path}")

    def test_historical_data_matches_claim(self):
        data = json.loads((ROOT / "references" / "case-data.json").read_text())
        baseline = data["pmc"]["baseline"]["mean_per_dispatch"]
        kperm = data["pmc"]["kperm"]["mean_per_dispatch"]
        self.assertEqual(baseline["SQ_INSTS_MFMA"], kperm["SQ_INSTS_MFMA"])
        self.assertEqual(baseline["SQ_LDS_BANK_CONFLICT"] / kperm["SQ_LDS_BANK_CONFLICT"], 25)
        for record in data["pmc"].values():
            samples = {}
            for row in record["rows"]:
                samples.setdefault(row["Counter_Name"], []).append(float(row["Counter_Value"]))
            for name, values in samples.items():
                self.assertEqual(sum(values) / len(values), record["mean_per_dispatch"][name])


if __name__ == "__main__":
    unittest.main()
