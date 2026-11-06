"""History cutoffs preserve the JSONL without making summarization requests."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from core.agent import Agent, HISTORY_CUTOFF_NOTICE, HISTORY_CUTOFF_STATE


FINAL = "<|final_answer|>done<\\final_answer|>"
ACTION = (
    "<|action|>python_code<\\action|>\n"
    "<|action_input|>\n```text\n_result_ = 'data'\n```\n<\\action_input|>"
)


class AgentHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.history = Path(directory.name) / "session.jsonl"
        self.backend = Mock()
        patcher = patch("core.agent.get_backend", return_value=self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.agent = self.make_agent()
        self.requests = []

    def make_agent(self, **kwargs):
        options = dict(
            max_tokens=500,
            max_steps=3,
            history=str(self.history),
            target_history_context_size=1,
        )
        options.update(kwargs)
        return Agent(**options)

    def notice(self, last_line, path=None):
        return HISTORY_CUTOFF_NOTICE.format(
            history_file=json.dumps(str((path or self.history).resolve()), ensure_ascii=False),
            line_range=f"lines 1-{last_line} (1-based, inclusive)" if last_line
                       else "no earlier lines (empty range)",
        )

    def append(self, content, state="user", role="user"):
        self.agent.save_history(role, content, state)
        self.agent.memory.append({"role": role, "content": content})

    def records(self):
        return [json.loads(line) for line in self.history.read_text().splitlines()]

    def respond_with(self, *responses):
        responses = iter(responses)

        def stream(inputs, max_tokens):
            self.requests.append([dict(message) for message in inputs])
            self.assertEqual(max_tokens, self.agent.max_tokens)
            return iter([next(responses)])

        self.backend.stream_response.side_effect = stream

    def test_cutoff_is_append_only_and_never_calls_model(self):
        system = {"role": "system", "content": "system " * 1000}
        self.agent.memory = [system]
        self.append("原文" * 501)
        original = self.history.read_bytes()

        self.assertTrue(self.agent.compact_history())
        self.backend.stream_response.assert_not_called()
        self.assertTrue(self.history.read_bytes().startswith(original))
        records = self.records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["content"], "原文" * 501)
        self.assertEqual(records[-1]["state"], HISTORY_CUTOFF_STATE)
        self.assertEqual(records[-1]["content"], self.notice(1))
        self.assertIn(str(self.history.resolve()), records[-1]["content"])
        self.assertIn("lines 1-1 (1-based, inclusive)", records[-1]["content"])
        self.assertEqual(
            self.agent.memory,
            [system, {"role": "user", "content": self.notice(1)}],
        )
        self.assertLess(self.agent.history_context_size(), 1)

    def test_target_is_strictly_exceeded_and_uses_characters_not_bytes(self):
        self.agent.memory = [{"role": "system", "content": "s" * 10000}]
        self.append("中" * 1000)
        original = self.history.read_bytes()
        self.assertEqual(self.agent.history_context_size(), 1)
        self.assertFalse(self.agent.compact_history())
        self.assertEqual(self.history.read_bytes(), original)
        self.append("字")
        self.assertTrue(self.agent.compact_history())

    def test_empty_history_does_not_create_cutoff(self):
        self.assertFalse(self.agent.compact_history())
        self.assertFalse(self.history.exists())

    def test_zero_disables_cutoffs(self):
        self.agent.target_history_context_size = 0
        self.append("x" * 10000)
        original = self.history.read_bytes()
        self.assertFalse(self.agent.compact_history())
        self.assertEqual(self.history.read_bytes(), original)
        self.backend.stream_response.assert_not_called()

    def test_negative_target_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_agent(target_history_context_size=-1)

    def test_disk_size_does_not_retrigger_cutoff(self):
        self.append("x" * 2000)
        self.assertTrue(self.agent.compact_history())
        self.assertGreater(self.history.stat().st_size, 1000)
        self.assertFalse(self.agent.compact_history())
        self.append("new task")
        self.assertFalse(self.agent.compact_history())
        self.assertEqual(len(self.records()), 3)

    def test_resume_uses_only_latest_cutoff_and_following_messages(self):
        self.append("old " * 300)
        self.assertTrue(self.agent.compact_history())
        self.append("middle " * 300)
        self.assertTrue(self.agent.compact_history())
        self.append("latest request")
        self.append("latest response", state="model", role="assistant")

        expected = [
            {"role": "user", "content": self.notice(3)},
            {"role": "user", "content": "latest request"},
            {"role": "assistant", "content": "latest response"},
        ]
        self.assertEqual(self.make_agent().load_history(), expected)
        self.assertEqual(self.agent.memory, expected)
        self.assertEqual(len(self.records()), 6)

    def test_loader_validates_cutoff_records_and_ignores_bad_lines(self):
        self.append("old")
        self.append(self.notice(1), state=HISTORY_CUTOFF_STATE)
        self.append("retained")
        self.append(self.notice(1))  # Text alone is not a boundary.
        self.append("not a boundary", state=HISTORY_CUTOFF_STATE, role="assistant")
        with self.history.open("a") as file:
            for record in (
                {"role": "system", "state": HISTORY_CUTOFF_STATE, "content": "bad role"},
                {"role": "user", "state": HISTORY_CUTOFF_STATE, "content": None},
                ["not a message"],
            ):
                file.write(json.dumps(record) + "\n")
            file.write("\n{invalid json}\n")
        self.append("tail", state="model", role="assistant")

        self.assertEqual(
            [item["content"] for item in self.agent.load_history()],
            [self.notice(1), "retained", self.notice(1), "not a boundary", "tail"],
        )

    def test_legacy_summary_remains_readable_without_new_summaries(self):
        self.append("old raw data")
        self.append("legacy summary", state="summary")
        self.append("next")
        self.assertEqual(
            [item["content"] for item in self.agent.load_history()],
            ["legacy summary", "next"],
        )
        self.append("x" * 1001)
        self.assertTrue(self.agent.compact_history())
        self.assertEqual(self.agent.load_history(), [
            {"role": "user", "content": self.notice(4)},
        ])
        self.assertEqual([r["state"] for r in self.records()].count("summary"), 1)
        self.backend.stream_response.assert_not_called()

    def test_missing_history_loads_empty(self):
        self.assertEqual(self.agent.load_history(), [])

    def test_unmarked_history_keeps_all_messages(self):
        self.append("request")
        self.append("response", state="model", role="assistant")
        self.assertEqual(self.agent.load_history(), self.agent.memory)

    def test_failed_history_read_does_not_clear_memory(self):
        self.append("x" * 1001)
        before = list(self.agent.memory)
        original = self.history.read_bytes()
        with patch.object(Path, "open", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.agent.compact_history()
        self.assertEqual(self.agent.memory, before)
        self.assertEqual(self.history.read_bytes(), original)
        self.backend.stream_response.assert_not_called()

    def test_failed_boundary_fsync_does_not_clear_memory(self):
        self.append("x" * 1001)
        before = list(self.agent.memory)
        with patch("core.agent.os.fsync", side_effect=OSError("fsync failure")):
            with self.assertRaisesRegex(OSError, "fsync failure"):
                self.agent.compact_history()
        self.assertEqual(self.agent.memory, before)
        self.backend.stream_response.assert_not_called()

    def test_oversized_initial_request_makes_only_normal_model_call(self):
        self.respond_with(FINAL)
        self.agent.run("task " * 300)

        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0][1:], [
            {"role": "user", "content": self.notice(1)},
        ])
        self.assertIn(str(self.history.resolve()), self.requests[0][0]["content"])
        records = self.records()
        self.assertEqual([r["state"] for r in records], [
            "user", HISTORY_CUTOFF_STATE, "model",
        ])
        self.assertEqual(records[0]["content"], "task " * 300)
        self.assertTrue(records[1]["run_id"])
        self.assertEqual(records[1]["run_id"], records[0]["run_id"])

    def test_tool_output_cutoff_and_subsequent_requests_share_active_segment(self):
        self.respond_with(ACTION, ACTION, FINAL)
        with patch("core.agent.ActionRunner.__call__", side_effect=["x" * 1200, "small"]):
            self.agent.run("task")

        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[0][-1]["content"], "task")
        self.assertEqual(self.requests[1][1:], [
            {"role": "user", "content": self.notice(3)},
        ])
        self.assertEqual(self.requests[2][1]["content"], self.notice(3))
        self.assertEqual(self.requests[2][2]["content"], ACTION)
        self.assertIn("small", self.requests[2][3]["content"])
        records = self.records()
        self.assertEqual([r["state"] for r in records].count(HISTORY_CUTOFF_STATE), 1)
        self.assertTrue(any("x" * 1200 in r["content"] for r in records))
        self.assertFalse(any(r["state"] == "summary" for r in records))
        self.assertEqual(self.agent.load_history(), self.agent.memory[1:])

    def test_resume_updates_history_path_after_work_file_moves(self):
        self.append("old " * 300)
        self.agent.compact_history()
        self.append("recent context")
        moved = self.history.with_name("resumed-work.jsonl")
        self.history.rename(moved)
        self.agent = self.make_agent(history=str(moved))
        self.respond_with(FINAL)
        self.agent.run("continue")

        self.assertEqual(len(self.requests), 1)
        self.assertIn(str(moved.resolve()), self.requests[0][0]["content"])
        self.assertNotIn(str(self.history), self.requests[0][0]["content"])
        self.assertEqual([m["content"] for m in self.requests[0][1:]], [
            self.notice(1, moved), "recent context", "continue",
        ])

    def test_max_steps_instruction_survives_cutoff(self):
        self.agent.max_steps = 1
        self.respond_with(ACTION, FINAL)
        with patch("core.agent.ActionRunner.__call__", return_value="x" * 1200):
            self.agent.run("task")

        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[-1][1]["content"], self.notice(3))
        self.assertIn("Reached max_steps=1", self.requests[-1][-1]["content"])
        self.assertEqual([r["state"] for r in self.records()][-3:], [
            HISTORY_CUTOFF_STATE, "max_steps", "max_steps_final",
        ])

    def test_failed_boundary_write_does_not_clear_memory(self):
        self.append("x" * 1001)
        before = list(self.agent.memory)
        original = self.history.read_bytes()
        real_open = Path.open

        def open_file(path, mode="r", *args, **kwargs):
            if mode == "ab+":
                raise OSError("append failure")
            return real_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", open_file):
            with self.assertRaisesRegex(OSError, "append failure"):
                self.agent.compact_history()
        self.assertEqual(self.agent.memory, before)
        self.assertEqual(self.history.read_bytes(), original)
        self.backend.stream_response.assert_not_called()

    def test_cutoff_range_counts_physical_lines_including_invalid_and_blank(self):
        # Embedded newlines in a JSON string are not extra physical file lines.
        record = json.dumps({"role": "user", "content": "large\n" * 300})
        self.history.write_text("\n{invalid json}\n" + record + "\n  \n")
        self.agent.memory = self.agent.load_history()
        original = self.history.read_bytes()

        self.assertTrue(self.agent.compact_history())
        self.assertTrue(self.history.read_bytes().startswith(original))
        lines = self.history.read_text().splitlines()
        self.assertEqual(len(lines), 5)
        marker = json.loads(lines[4])
        self.assertEqual(marker["content"], self.notice(4))
        self.assertIn("lines 1-4 (1-based, inclusive)", marker["content"])
        self.assertEqual(self.make_agent().load_history(), self.agent.memory)

    def test_cutoff_after_unterminated_jsonl_has_correct_range(self):
        record = {"role": "user", "content": "中文\n" * 400}
        original = json.dumps(record, ensure_ascii=False).encode("utf-8")
        self.history.write_bytes(original)
        self.agent.memory = self.agent.load_history()

        self.assertTrue(self.agent.compact_history())
        self.assertTrue(self.history.read_bytes().startswith(original + b"\n"))
        records = self.records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], record)
        self.assertEqual(records[1]["content"], self.notice(1))
        self.assertEqual(self.make_agent().load_history(), self.agent.memory)

    def test_crlf_history_uses_same_line_range_when_created_and_reloaded(self):
        record = json.dumps({"role": "user", "content": "x" * 1001}).encode()
        self.history.write_bytes(b"\r\n" + record + b"\r\n")
        self.agent.memory = self.agent.load_history()
        self.assertTrue(self.agent.compact_history())
        self.assertEqual(self.agent.memory[0]["content"], self.notice(2))
        self.assertEqual(self.make_agent().load_history(), self.agent.memory)

    def test_loader_refreshes_legacy_notice_using_current_position_without_rewriting(self):
        self.append("old")
        self.append("more old")
        self.append("[History cutoff] Earlier history is not loaded.", state=HISTORY_CUTOFF_STATE)
        self.append("tail")
        original = self.history.read_bytes()
        loaded = self.agent.load_history()
        self.assertEqual(loaded[0]["content"], self.notice(2))
        self.assertEqual(loaded[1]["content"], "tail")
        self.assertEqual(self.history.read_bytes(), original)

    def test_cutoff_at_first_line_reports_empty_range(self):
        self.append("old notice", state=HISTORY_CUTOFF_STATE)
        self.assertEqual(self.agent.load_history(), [
            {"role": "user", "content": self.notice(0)},
        ])

    def test_relative_unicode_path_is_absolute_in_notice(self):
        self.agent = self.make_agent(history="历史 folder/../会话")
        notice = self.agent.history_cutoff_notice(42)
        path = self.agent.history_file.resolve()
        self.assertTrue(path.is_absolute())
        self.assertIn(json.dumps(str(path), ensure_ascii=False), notice)
        self.assertIn("lines 1-42 (1-based, inclusive)", notice)


if __name__ == "__main__":
    unittest.main()
