"""Durable history publication belongs to the server, not core/agent.py."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import server


class HistoryCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.temp = Path(directory.name)
        patcher = patch.object(server, "TEMP_DIR", self.temp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.history = self.temp / "session.jsonl"
        self.work = self.temp / "working.jsonl"
        self.original = self.record("original")
        self.history.write_bytes(self.original)
        self.checkpoint = server.HistoryCheckpoint("session.jsonl", self.work)
        self.checkpoint.prepare()

    @staticmethod
    def record(content):
        return (json.dumps(dict(role="user", content=content), ensure_ascii=False) + "\n").encode()

    def append(self, data):
        with self.work.open("ab") as file:
            file.write(data)

    def test_publish_waits_for_complete_json_utf8_and_newline(self):
        record = self.record("中文")
        split = record.index("中".encode()) + 1
        self.append(record[:split])
        self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original)
        self.append(record[split:-1])
        self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original)
        self.append(record[-1:])
        self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original + record)
        self.assertEqual(self.checkpoint.pending_bytes, 0)

    def test_failed_fsync_never_replaces_committed_history(self):
        self.append(self.record("next"))
        with patch.object(server.os, "fsync", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original)
        self.assertEqual(self.checkpoint.published, len(self.original))
        self.assertEqual(list(self.temp.glob(".agent-publish-*")), [])
        self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original + self.record("next"))

    def test_failed_rename_never_replaces_committed_history(self):
        self.append(self.record("next"))
        with patch.object(server.os, "replace", side_effect=OSError("rename failure")):
            with self.assertRaises(OSError):
                self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original)
        self.assertEqual(list(self.temp.glob(".agent-publish-*")), [])

    def test_file_is_fsynced_before_rename_and_directory_afterwards(self):
        self.append(self.record("next"))
        calls = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            calls.append("fsync")
            return real_fsync(fd)

        def replace(source, target):
            self.assertEqual(self.history.read_bytes(), self.original)
            self.assertEqual(Path(source).read_bytes(), self.original + self.record("next"))
            calls.append("replace")
            return real_replace(source, target)

        with patch.object(server.os, "fsync", side_effect=fsync), patch.object(
            server.os, "replace", side_effect=replace
        ):
            self.checkpoint.commit()
        self.assertEqual(calls, ["fsync", "replace", "fsync"] if os.name == "posix" else ["fsync", "replace"])

    def test_concurrent_readers_only_see_complete_snapshots(self):
        stop = threading.Event()
        failures = []
        ready = threading.Event()

        def read():
            try:
                while not stop.is_set():
                    raw = self.history.read_bytes()
                    self.assertTrue(raw.endswith(b"\n"))
                    records = [json.loads(line) for line in raw.splitlines()]
                    self.assertEqual(records[0]["content"], "original")
                    ready.set()
            except BaseException as exc:
                failures.append(exc)

        reader = threading.Thread(target=read)
        reader.start()
        try:
            self.assertTrue(ready.wait(5))
            for index in range(15):
                record = self.record(f"checkpoint {index} " + "数据" * 1000)
                self.append(record[:-1])
                self.checkpoint.commit()
                self.append(b"\n")
                self.checkpoint.commit()
        finally:
            stop.set()
            reader.join()
        self.assertFalse(failures, failures)
        self.assertEqual(len(self.history.read_bytes().splitlines()), 16)

    def test_complete_malformed_record_is_rejected_not_silently_skipped(self):
        self.append(b'{"role":"assistant","content": broken}\n')
        with self.assertRaisesRegex(ValueError, "不会忽略"):
            self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original)

    def test_external_history_changes_are_not_overwritten(self):
        self.append(self.record("agent result"))
        external = self.record("external change")
        self.history.write_bytes(external)
        with self.assertRaisesRegex(ValueError, "其他进程修改"):
            self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), external)

    def test_valid_import_without_final_newline_is_normalized_only_in_work_copy(self):
        self.history.write_bytes(self.original[:-1])
        self.checkpoint = server.HistoryCheckpoint("session.jsonl", self.work)
        self.checkpoint.prepare()
        self.assertEqual(self.history.read_bytes(), self.original[:-1])
        self.assertEqual(self.work.read_bytes(), self.original)
        self.append(self.record("next"))
        self.checkpoint.commit()
        self.assertEqual(self.history.read_bytes(), self.original + self.record("next"))


    def test_discard_fully_committed_copy_preserves_formal_history(self):
        record = self.record("final")
        self.append(record)
        self.checkpoint.commit()
        self.checkpoint.discard_committed_work_copy()
        self.assertFalse(self.work.exists())
        self.assertEqual(self.history.read_bytes(), self.original + record)

    def test_discard_handles_empty_committed_history(self):
        self.history.unlink()
        self.checkpoint = server.HistoryCheckpoint("session.jsonl", self.work)
        self.checkpoint.prepare()
        self.checkpoint.commit()
        self.checkpoint.discard_committed_work_copy()
        self.assertFalse(self.work.exists())
        self.assertEqual(self.history.read_bytes(), b"")

    def test_discard_refuses_unpublished_partial_record(self):
        self.append(b'{"role":"user","content":"partial')
        self.checkpoint.commit()
        with self.assertRaises(ValueError):
            self.checkpoint.discard_committed_work_copy()
        self.assertTrue(self.work.is_file())
        self.assertEqual(self.history.read_bytes(), self.original)

    def test_discard_refuses_missing_or_changed_formal_history(self):
        self.checkpoint.commit()
        for replacement in (None, self.record("external")):
            if replacement is None:
                self.history.unlink()
            else:
                self.history.write_bytes(replacement)
            with self.assertRaises(ValueError):
                self.checkpoint.discard_committed_work_copy()
            self.assertTrue(self.work.is_file())

    def test_discard_compares_bytes_not_only_size(self):
        self.checkpoint.commit()
        changed = self.original.replace(b"original", b"modified")
        self.assertEqual(len(changed), len(self.original))
        self.work.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "内容不同"):
            self.checkpoint.discard_committed_work_copy()
        self.assertEqual(self.work.read_bytes(), changed)
        self.assertEqual(self.history.read_bytes(), self.original)

    def test_discard_checks_for_uncommitted_appends(self):
        self.checkpoint.commit()
        self.append(self.record("late data"))
        with self.assertRaises(ValueError):
            self.checkpoint.discard_committed_work_copy()
        self.assertTrue(self.work.is_file())

    @unittest.skipUnless(os.name == "posix", "Requires POSIX O_NOFOLLOW")
    def test_discard_does_not_follow_work_symlink(self):
        self.checkpoint.commit()
        self.work.unlink()
        self.work.symlink_to(self.history)
        with self.assertRaises(OSError):
            self.checkpoint.discard_committed_work_copy()
        self.assertTrue(self.work.is_symlink())
        self.assertEqual(self.history.read_bytes(), self.original)

    def test_discard_large_history_uses_bounded_reads(self):
        record = self.record("a" * (2 * 1024 * 1024 + 17))
        self.append(record)
        self.checkpoint.commit()
        self.checkpoint.discard_committed_work_copy()
        self.assertFalse(self.work.exists())
        self.assertEqual(self.history.read_bytes(), self.original + record)


if __name__ == "__main__":
    unittest.main()
