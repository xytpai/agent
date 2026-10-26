"""History deletion regression tests; only isolated temporary files are deleted."""
import json
import os
from pathlib import Path
import secrets
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

import server


class HistoryDeleteTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.temp = self.root / "temp"
        self.temp.mkdir()
        for name, value in (("TEMP_DIR", self.temp), ("LOG_PATH", self.root / "log")):
            mocker = patch.object(server, name, value)
            mocker.start()
            self.addCleanup(mocker.stop)
        original = dict(server.app.config)
        self.addCleanup(lambda: server.app.config.update(original))
        server.app.config.update(TESTING=True, AUTH_USERNAME="test", AUTH_PASSWORD_HASH="configured")
        self.sid = secrets.token_hex(16)
        server.AUTH_SESSIONS[self.sid] = time.monotonic() + 300
        self.addCleanup(server.AUTH_SESSIONS.pop, self.sid, None)
        self.client = server.app.test_client()
        with self.client.session_transaction(base_url="https://localhost") as session:
            session["auth_id"] = self.sid
            session["csrf"] = "delete-test-csrf"
        self.path = self.temp / "session.jsonl"
        self.original = b'{"role":"user","content":"saved task"}\n'
        self.path.write_bytes(self.original)

    def get(self, path, **kwargs):
        return self.client.get(path, base_url="https://localhost", **kwargs)

    def body(self, path=None):
        path = path or self.path
        return dict(name=path.name, confirm_name=path.name,
                    version=server.history_version(path.lstat()))

    def delete(self, body=None, **kwargs):
        return self.client.post(
            "/api/histories/delete",
            json=self.body() if body is None else body,
            base_url="https://localhost",
            headers={"X-CSRF-Token": "delete-test-csrf"},
            **kwargs,
        )

    def test_confirmed_delete_removes_only_formal_history(self):
        other = self.temp / "other.jsonl"
        other.write_bytes(self.original)
        runs = self.temp / "web-runs"
        runs.mkdir()
        artifacts = [runs / ("a" * 32 + suffix) for suffix in (".json", ".events", ".history.jsonl")]
        artifacts.append(server.LOG_PATH)
        for path in artifacts:
            path.write_bytes(b"retained")
        preview = self.get("/api/histories/preview?name=session.jsonl")
        self.assertEqual(preview.status_code, 200)
        body = self.body()
        self.assertEqual(body["version"], preview.json["file"]["version"])
        listing = self.get("/api/histories").json["files"]
        self.assertEqual(next(f["version"] for f in listing if f["name"] == self.path.name), body["version"])
        response = self.delete(body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["deleted"], self.path.name)
        self.assertFalse(self.path.exists())
        self.assertEqual(other.read_bytes(), self.original)
        for path in artifacts:
            self.assertEqual(path.read_bytes(), b"retained")
        self.assertEqual([f["name"] for f in self.get("/api/histories").json["files"]], ["other.jsonl"])
        self.assertEqual(self.get("/api/histories/preview?name=session.jsonl").status_code, 404)
        self.assertFalse(server.RUN_LOCK.locked())

    def test_unicode_spaces_and_empty_history(self):
        path = self.temp / " 中文 测试.jsonl"
        path.touch()
        response = self.delete(self.body(path))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(path.exists())

    def test_missing_or_wrong_confirmation_never_deletes(self):
        for confirmation in (None, "", "wrong.jsonl", "session", " session.jsonl", True, ["session.jsonl"]):
            with self.subTest(confirmation=confirmation):
                body = self.body()
                body["confirm_name"] = confirmation
                self.assertEqual(self.delete(body).status_code, 400)
                self.assertEqual(self.path.read_bytes(), self.original)

    def test_missing_or_invalid_version_never_deletes(self):
        for version in (None, "", "x" * 64, 123, "a" * 63):
            with self.subTest(version=version):
                body = self.body()
                body["version"] = version
                self.assertEqual(self.delete(body).status_code, 400)
                self.assertTrue(self.path.exists())

    def test_authentication_required(self):
        anonymous = server.app.test_client()
        response = anonymous.post("/api/histories/delete", json=self.body(), base_url="https://localhost")
        self.assertEqual(response.status_code, 401)
        self.assertTrue(self.path.exists())

    def test_csrf_required(self):
        for headers in ({}, {"X-CSRF-Token": "incorrect"}):
            response = self.client.post("/api/histories/delete", json=self.body(),
                                        base_url="https://localhost", headers=headers)
            self.assertEqual(response.status_code, 403)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_get_cannot_delete(self):
        response = self.get("/api/histories/delete?name=session.jsonl&confirm_name=session.jsonl")
        self.assertEqual(response.status_code, 405)
        self.assertTrue(self.path.exists())

    def test_bad_names_and_nested_paths_rejected(self):
        names = ["../session.jsonl", "/tmp/session.jsonl", "..\\session.jsonl",
                 "web-runs/a.history.jsonl", "log", ".", "..", "", "x\x00.jsonl",
                 "a:b.jsonl", None, [], 10, "\ud800.jsonl"]
        for name in names:
            with self.subTest(name=repr(name)):
                self.assertEqual(self.delete(dict(name=name, confirm_name=name, version="a" * 64)).status_code, 400)
        self.assertTrue(self.path.exists())
        self.assertFalse(server.RUN_LOCK.locked())

    def test_missing_file_and_repeat_delete(self):
        body = self.body()
        self.assertEqual(self.delete(body).status_code, 200)
        self.assertEqual(self.delete(body).status_code, 404)
        self.assertFalse(server.RUN_LOCK.locked())

    def test_stale_version_after_append(self):
        body = self.body()
        with self.path.open("ab") as stream:
            stream.write(self.original)
        self.assertEqual(self.delete(body).status_code, 409)
        self.assertEqual(self.path.read_bytes(), self.original * 2)
        self.assertEqual(self.delete().status_code, 200)

    def test_same_size_replacement_requires_new_confirmation(self):
        body = self.body()
        replacement = self.temp / "replacement"
        replacement.write_bytes(self.original)
        replacement.replace(self.path)
        self.assertEqual(self.delete(body).status_code, 409)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_symlink_and_dangling_symlink_rejected(self):
        outside = self.root / "outside.jsonl"
        outside.write_bytes(b"outside")
        for target in (outside, self.root / "missing.jsonl"):
            link = self.temp / "link.jsonl"
            link.symlink_to(target)
            self.assertEqual(self.delete(self.body(link)).status_code, 400)
            self.assertTrue(link.is_symlink())
            link.unlink()
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_directory_rejected(self):
        directory = self.temp / "directory.jsonl"
        directory.mkdir()
        self.assertEqual(self.delete(self.body(directory)).status_code, 400)
        self.assertTrue(directory.is_dir())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO needs POSIX")
    def test_fifo_rejected_without_opening_it(self):
        fifo = self.temp / "pipe.jsonl"
        os.mkfifo(fifo)
        self.assertEqual(self.delete(self.body(fifo)).status_code, 400)
        self.assertTrue(fifo.exists())

    @unittest.skipUnless(os.name == "posix", "dir_fd behavior needs POSIX")
    def test_temp_directory_symlink_not_followed(self):
        target = self.root / "linked-temp"
        target.symlink_to(self.temp, target_is_directory=True)
        body = self.body()
        with patch.object(server, "TEMP_DIR", target), patch.object(server.app.logger, "exception"):
            self.assertEqual(self.delete(body).status_code, 500)
        self.assertTrue(self.path.exists())

    def test_running_starting_or_finalizing_task_blocks_all_deletion(self):
        self.assertTrue(server.RUN_LOCK.acquire(blocking=False))
        try:
            # ACTIVE_RUN can be absent during startup or final cleanup.
            for current in (None, SimpleNamespace(history="session.jsonl"),
                            SimpleNamespace(history="other.jsonl")):
                with patch.object(server, "ACTIVE_RUN", current):
                    self.assertEqual(self.delete().status_code, 409)
                    self.assertEqual(self.path.read_bytes(), self.original)
                    self.assertTrue(server.RUN_LOCK.locked())
        finally:
            server.RUN_LOCK.release()
        self.assertEqual(self.delete().status_code, 200)

    def test_deletion_holds_startup_lock(self):
        original_unlink = server.os.unlink
        def checked_unlink(*args, **kwargs):
            self.assertTrue(server.RUN_LOCK.locked())
            self.assertFalse(server.RUN_LOCK.acquire(blocking=False))
            return original_unlink(*args, **kwargs)
        with patch.object(server.os, "unlink", side_effect=checked_unlink):
            self.assertEqual(self.delete().status_code, 200)
        self.assertFalse(server.RUN_LOCK.locked())

    def test_permission_error_releases_lock_and_preserves_file(self):
        with patch.object(server.os, "unlink", side_effect=PermissionError("test")), \
                patch.object(server.app.logger, "exception"):
            self.assertEqual(self.delete().status_code, 500)
        self.assertFalse(server.RUN_LOCK.locked())
        self.assertEqual(self.path.read_bytes(), self.original)

    @unittest.skipUnless(os.name == "posix", "directory fsync needs POSIX")
    def test_fsync_failure_does_not_claim_file_still_exists(self):
        with patch.object(server.os, "fsync", side_effect=OSError("test")), \
                patch.object(server.app.logger, "warning"):
            response = self.delete()
        self.assertEqual(response.status_code, 200)
        self.assertIn("文件已删除", response.json["warning"])
        self.assertFalse(self.path.exists())
        self.assertFalse(server.RUN_LOCK.locked())

    def test_invalid_json_shape_and_form_submission(self):
        for value in ([], "name", 12, None):
            response = self.client.post(
                "/api/histories/delete", data=json.dumps(value), content_type="application/json",
                headers={"X-CSRF-Token": "delete-test-csrf"}, base_url="https://localhost",
            )
            self.assertEqual(response.status_code, 400)
        self.assertTrue(self.path.exists())
        response = self.client.post(
            "/api/histories/delete", data=self.body(),
            headers={"X-CSRF-Token": "delete-test-csrf"}, base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
