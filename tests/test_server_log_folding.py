"""Run the actual embedded log renderer in Node with a minimal inert DOM."""
import ast
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LogFoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        node = shutil.which("node") or shutil.which("nodejs")
        if not node:
            try:
                from playwright._impl._driver import compute_driver_executable
                candidate, _ = compute_driver_executable()
                if Path(candidate).is_file():
                    node = candidate
            except ImportError:
                pass
        if not node:
            raise unittest.SkipTest("JavaScript tests require Node or Playwright's bundled Node")
        cls.node = str(node)
        tree = ast.parse((ROOT / "server.py").read_text())
        cls.page = next(
            ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PAGE" for target in n.targets)
        )
        cls.script = cls.page.split("<script>", 1)[1].split("</script>", 1)[0]

    def test_full_page_javascript_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.js"
            path.write_text(self.script)
            result = subprocess.run([self.node, "--check", str(path)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_streaming_folding_behaviors(self):
        renderer = self.script.split("// BEGIN LOG HIGHLIGHTER", 1)[1].split("// END LOG HIGHLIGHTER", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renderer.js"
            path.write_text(renderer)
            result = subprocess.run(
                [self.node, str(ROOT / "tests/log_folding.cjs"), str(path)],
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout)

    def test_only_main_log_uses_folding_and_copy_is_raw(self):
        self.assertIn('<div id="out" hidden></div>', self.page)
        self.assertIn("new FoldedLogHighlighter(out, MAX_DISPLAY)", self.script)
        self.assertIn("new LogHighlighter(content,", self.script)
        self.assertIn("rawLog = logHighlighter.text", self.script)
        self.assertIn("navigator.clipboard.writeText(rawLog)", self.script)
        self.assertIn('apiFetch("/log"', self.script)


    def test_history_loading_clears_stale_live_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.js"
            path.write_text(self.script)
            result = subprocess.run(
                [self.node, str(ROOT / "tests/history_load.cjs"), str(path)],
                capture_output=True, text=True, timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout)


if __name__ == "__main__":
    unittest.main()
