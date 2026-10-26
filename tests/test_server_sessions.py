"""Session recovery tests; no model gateway or real tools are used."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import server


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.temp = Path(self.directory.name)
        self.original_config = dict(server.app.config)
        server.app.config.update(TESTING=True, AUTH_USERNAME="test", AUTH_PASSWORD_HASH="configured")
        self.addCleanup(lambda: server.app.config.update(self.original_config))
        for name, value in (("TEMP_DIR", self.temp), ("LOG_PATH", self.temp / "log")):
            patcher = patch.object(server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = server.app.test_client()
        server.AUTH_SESSIONS["test-session"] = time.monotonic() + 300
        with self.client.session_transaction(base_url="https://localhost") as session:
            session["auth_id"] = "test-session"
            session["csrf"] = "test-csrf"
        self.addCleanup(self.clean_run)

    def clean_run(self):
        current = server.ACTIVE_RUN
        if current:
            current.stop()
            self.wait_closed(current)
        server.AUTH_SESSIONS.pop("test-session", None)

    def get(self, path, **kwargs):
        return self.client.get(path, base_url="https://localhost", **kwargs)

    def post(self, path, data=None, **kwargs):
        return self.client.post(path, data=data, base_url="https://localhost",
                                headers={"X-CSRF-Token": "test-csrf"}, **kwargs)

    def wait_closed(self, current):
        deadline = time.monotonic() + 5
        while not current.closed or server.RUN_LOCK.locked():
            if time.monotonic() > deadline:
                self.fail("background collector did not release the run")
            time.sleep(0.01)
        worker = getattr(current, "worker", None)
        if worker is not None:
            worker.join(timeout=5)

    def start(self, script, **fields):
        real_popen = subprocess.Popen

        def fake_agent(command, **kwargs):
            self.command = command
            self.input_text = Path(command[command.index("--input") + 1]).read_text()
            self.work_path = Path(command[command.index("--history") + 1])
            kwargs["env"]["TEST_HISTORY"] = str(self.work_path)
            return real_popen([sys.executable, "-u", "-c", script], **kwargs)

        data = dict(input="task", history="session", model="fake-model", effort="high")
        data.update(fields)
        with patch.object(server.subprocess, "Popen", side_effect=fake_agent):
            response = self.post("/run", data, buffered=False)
        return response

    @staticmethod
    def events(response):
        return [json.loads(line) for line in response.data.splitlines()]

    def test_disconnect_reconnect_stop_and_resume(self):
        history = self.temp / "session.jsonl"
        original = json.dumps({"role": "user", "content": "original task"}) + "\n"
        history.write_text(original)
        response = self.start("import time; print('准备好了'); time.sleep(60)")
        first = json.loads(next(iter(response.response)))
        current = server.ACTIVE_RUN
        response.close()  # Closing the only HTTP subscriber must not kill it.
        self.assertFalse(current.closed)
        self.assertIsNone(current.process.poll())
        self.assertTrue(server.RUN_LOCK.locked())
        latest = self.get("/api/runs/latest").json["run"]
        self.assertEqual(latest["id"], first["id"])
        self.assertEqual(latest["history"], "session.jsonl")
        self.assertEqual(latest["status"], "running")
        self.assertEqual(self.post("/run", dict(input="new", model="fake-model")).status_code, 409)
        self.assertEqual(self.post("/stop", {"id": current.id}).status_code, 200)
        self.wait_closed(current)
        events = self.events(self.get(f"/api/runs/{current.id}/events?offset={first['offset']}"))
        self.assertTrue(events[-1]["stopped"])
        self.assertNotIn("start", [item["type"] for item in events])
        for item in events:
            self.assertGreaterEqual(item["offset"], first["offset"])
        self.assertEqual(history.read_text(), original)

        resumed = self.start("print('continued')", resume="1", input="")
        events = self.events(resumed)
        resumed.close()
        self.assertEqual(events[0]["history"], "session.jsonl")
        self.assertIn("outcome is unknown", self.input_text)
        self.assertEqual(events[-1]["code"], 0)
        self.assertFalse(server.RUN_LOCK.locked())

    def test_completed_run_survives_loss_of_in_memory_state(self):
        response = self.start("print('hello 世界')")
        events = self.events(response)
        response.close()
        run_id = events[0]["id"]
        self.assertIsNone(server.ACTIVE_RUN)
        data = self.get("/api/runs/latest?history=session").json["run"]
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["model"], "fake-model")
        self.assertEqual(data["effort"], "high")
        replay = self.events(self.get(f"/api/runs/{run_id}/events"))
        self.assertEqual("".join(e["text"] for e in replay if e["type"] == "output"), "hello 世界\n")
        self.assertEqual(replay[-1]["code"], 0)
        self.assertIsNone(self.get("/api/runs/latest?history=another").json["run"])

    def test_orphaned_run_and_partial_event_are_recoverable(self):
        current = server.AgentRun("task", "old.jsonl", "model", "low", 12, 200)
        server.run_path(current.id, "json").parent.mkdir()
        current.save_metadata()
        current.emit("start", **current.metadata)
        current.emit("output", text="saved output")
        with server.run_path(current.id, "events").open("ab") as file:
            file.write(b'{"type":"output","text":"incomplete')
        latest = self.get("/api/runs/latest").json["run"]
        self.assertEqual(latest["status"], "interrupted")
        events = self.events(self.get(f"/api/runs/{current.id}/events"))
        self.assertEqual(events[1]["text"], "saved output")
        self.assertTrue(events[-1]["interrupted"])
        self.assertFalse(server.RUN_LOCK.locked())

    def test_validation_and_auth(self):
        for history in ("../escape", "/tmp/escape", "not-existing"):
            response = self.post("/run", dict(input="", history=history, model="fake", resume="1"))
            self.assertEqual(response.status_code, 400)
        (self.temp / "empty.jsonl").touch()
        self.assertEqual(self.post("/run", dict(input="", history="empty", model="fake", resume="1")).status_code, 400)
        (self.temp / "linked.jsonl").symlink_to(self.temp / "empty.jsonl")
        self.assertEqual(self.post("/run", dict(input="x", history="linked", model="fake")).status_code, 400)
        self.assertEqual(self.post("/run", dict(input="", model="fake")).status_code, 400)
        self.assertEqual(self.get("/api/runs/bad/events").status_code, 400)
        self.assertEqual(self.get("/api/runs/" + "a" * 32 + "/events").status_code, 404)
        anonymous = server.app.test_client()
        self.assertEqual(anonymous.get("/api/runs/latest").status_code, 401)
        self.assertEqual(anonymous.get("/api/runs/" + "a" * 32 + "/events").status_code, 401)
        self.assertEqual(self.client.post("/run", base_url="https://localhost").status_code, 403)

    def test_event_cursor_boundaries_and_failed_process(self):
        response = self.start("print('failure'); raise SystemExit(7)")
        events = self.events(response)
        response.close()
        run_id = events[0]["id"]
        self.assertEqual(events[-1]["code"], 7)
        for offset in ("-1", "1", "999999999", "abc"):
            self.assertEqual(self.get(f"/api/runs/{run_id}/events?offset={offset}").status_code, 400)
        replay = self.events(self.get(f"/api/runs/{run_id}/events?offset={events[-1]['offset']}"))
        self.assertEqual(replay[-1]["code"], 7)

    def test_start_failure_releases_lock(self):
        with patch.object(server.subprocess, "Popen", side_effect=OSError("test failure")):
            response = self.post("/run", dict(input="task", model="fake"))
        self.assertEqual(response.status_code, 500)
        self.assertFalse(server.RUN_LOCK.locked())
        self.assertIsNone(server.ACTIVE_RUN)
        self.assertEqual(list(self.temp.glob("web-input-*")), [])

    def test_resume_with_additional_instructions_keeps_interruption_warning(self):
        (self.temp / "session.jsonl").write_text('{"role":"user","content":"old task"}\n')
        response = self.start("print('ok')", resume="1", input="inspect the files first")
        self.events(response)
        response.close()
        self.assertIn("outcome is unknown", self.input_text)
        self.assertIn("Additional instructions:\ninspect the files first", self.input_text)
        self.assertEqual(self.work_path.parent, self.temp / "web-runs")
        self.assertNotEqual(self.work_path, self.temp / "session.jsonl")

    def test_normal_shutdown_stops_and_persists_run(self):
        response = self.start("import time; time.sleep(60)")
        current = server.ACTIVE_RUN
        response.close()
        server.shutdown_active_run()
        self.assertTrue(current.closed)
        self.assertFalse(server.RUN_LOCK.locked())
        self.assertEqual(server.read_run(current.id)["status"], "stopped")
        self.assertTrue(self.work_path.is_file())

    def test_stop_during_partial_write_never_publishes_broken_record(self):
        original = b'{"role":"user","content":"original"}\n'
        complete = b'{"role":"assistant","content":"complete answer"}\n'
        partial = b'{"role":"user","content":"' + bytes([0xe4, 0xb8])
        history = self.temp / "session.jsonl"
        history.write_bytes(original)
        script = (
            "import os, time; "
            "f=open(os.environ['TEST_HISTORY'], 'ab', buffering=0); "
            f"f.write({complete!r}); f.write({partial!r}); "
            "print('mid-write', flush=True); time.sleep(60)"
        )
        response = self.start(script)
        current = server.ACTIVE_RUN
        for raw in response.response:
            if "mid-write" in json.loads(raw).get("text", ""):
                break
        response.close()
        self.assertEqual(self.post("/stop", {"id": current.id}).status_code, 200)
        self.wait_closed(current)
        self.assertEqual(history.read_bytes(), original + complete)
        self.assertEqual(self.work_path.read_bytes(), original + complete + partial)
        status = server.read_run(current.id)
        self.assertEqual(status["history_pending_bytes"], len(partial))
        self.assertEqual(status["history_committed_bytes"], len(original + complete))
        self.assertIn("未提交", status["history_warning"])
        self.assertEqual(status["status"], "stopped")

        resumed_record = b'{"role":"user","content":"resumed"}\n'
        response = self.start(
            "import os; f=open(os.environ['TEST_HISTORY'], 'ab'); "
            f"f.write({resumed_record!r}); f.close()", resume="1", input="",
        )
        events = self.events(response)
        response.close()
        self.assertEqual(events[-1]["code"], 0)
        self.assertEqual(len(history.read_bytes().splitlines()), 3)
        self.assertTrue(all(json.loads(line) for line in history.read_bytes().splitlines()))

    def test_checkpoint_progress_is_saved_even_without_stdout(self):
        history = self.temp / "session.jsonl"
        record = b'{"role":"user","content":"saved"}\n'
        response = self.start(
            "import os, time; "
            "f=open(os.environ['TEST_HISTORY'], 'ab', buffering=0); "
            f"f.write({record!r}); time.sleep(60)"
        )
        current = server.ACTIVE_RUN
        response.close()
        deadline = time.monotonic() + 5
        while not history.exists() or not history.read_bytes():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertEqual(json.loads(history.read_bytes()), dict(role="user", content="saved"))
        self.assertIsNone(current.process.poll())
        self.assertNotEqual(self.work_path, history)

    def test_invalid_existing_history_rejected_without_changing_it(self):
        history = self.temp / "session.jsonl"
        for original in (
            b'{"role":"user","content":"ok"}\n{"role":"assistant","content":"broken',
            b'{"role":"user","content":"' + bytes([0xe4, 0xb8]),
        ):
            history.write_bytes(original)
            with patch.object(server.subprocess, "Popen") as spawn:
                response = self.post("/run", dict(model="fake", resume="1", history="session"))
            self.assertEqual(response.status_code, 400)
            spawn.assert_not_called()
            self.assertEqual(history.read_bytes(), original)
            self.assertFalse(server.RUN_LOCK.locked())

    def test_unmodified_agent_loads_and_saves_through_working_copy(self):
        original = b'{"role":"user","content":"original task"}\n'
        history = self.temp / "session.jsonl"
        history.write_bytes(original)
        real_popen = subprocess.Popen

        def simulated_model(command, **kwargs):
            script = (
                "import runpy, sys, core.backends\n"
                "class Backend:\n"
                "    def stream_response(self, messages, max_tokens):\n"
                "        assert messages[1]['content'] == 'original task'\n"
                "        assert 'outcome is unknown' in messages[-1]['content']\n"
                "        yield '<|final_answer|>resumed<\\\\final_answer|>'\n"
                "core.backends.get_backend = lambda **kwargs: Backend()\n"
                f"sys.argv = {command[1:]!r}\n"
                "runpy.run_module('core.agent', run_name='__main__')\n"
            )
            return real_popen([sys.executable, "-u", "-c", script], **kwargs)

        with patch.object(server.subprocess, "Popen", side_effect=simulated_model):
            response = self.post("/run", dict(
                model="fake", resume="1", history="session", max_steps="1",
            ))
            events = self.events(response)
        response.close()
        self.assertEqual(events[-1]["code"], 0, events)
        records = [json.loads(line) for line in history.read_bytes().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["content"], "original task")
        self.assertEqual(records[-1]["state"], "model")
        self.assertIn("resumed", records[-1]["content"])
        self.assertTrue(history.read_bytes().startswith(original))

    def test_logout_still_stops_owned_run(self):
        response = self.start("import time; time.sleep(60)")
        current = server.ACTIVE_RUN
        response.close()
        self.assertEqual(self.post("/logout").status_code, 200)
        self.wait_closed(current)
        self.assertTrue(current.stopped)
        self.assertTrue(self.work_path.is_file())


    def test_success_removes_only_own_copy_and_can_replay_and_resume(self):
        history = self.temp / "session.jsonl"
        original = b'{"role":"user","content":"original"}\n'
        added = b'{"role":"assistant","content":"first answer"}\n'
        history.write_bytes(original)
        runs = self.temp / "web-runs"
        runs.mkdir()
        old_copy = runs / ("a" * 32 + ".history.jsonl")
        old_copy.write_bytes(b"old copy retained")
        response = self.start(
            "import os; f=open(os.environ['TEST_HISTORY'],'ab'); "
            f"f.write({added!r}); f.close(); print('first output')"
        )
        events = self.events(response)
        response.close()
        run_id = events[0]["id"]
        self.assertEqual(events[-1]["code"], 0)
        self.assertFalse(self.work_path.exists())
        self.assertEqual(history.read_bytes(), original + added)
        self.assertEqual(old_copy.read_bytes(), b"old copy retained")
        for suffix in ("json", "events"):
            self.assertTrue(server.run_path(run_id, suffix).is_file())
        self.assertEqual(server.LOG_PATH.read_text(), "first output\n")
        metadata = self.get("/api/runs/latest").json["run"]
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["history_pending_bytes"], 0)
        self.assertEqual(metadata["history_committed_bytes"], len(original + added))
        replay = self.events(self.get(f"/api/runs/{run_id}/events"))
        self.assertEqual("".join(e["text"] for e in replay if e["type"] == "output"), "first output\n")
        self.assertEqual(replay[-1]["code"], 0)

        more = b'{"role":"assistant","content":"continued answer"}\n'
        response = self.start(
            "import os; from pathlib import Path; p=Path(os.environ['TEST_HISTORY']); "
            f"assert p.read_bytes() == {original + added!r}; "
            f"f=p.open('ab'); f.write({more!r}); f.close(); print('continued')",
            resume="1", input="",
        )
        resumed = self.events(response)
        response.close()
        self.assertNotEqual(resumed[0]["id"], run_id)
        self.assertEqual(resumed[-1]["code"], 0)
        self.assertEqual(history.read_bytes(), original + added + more)
        self.assertFalse(self.work_path.exists())

    def test_successful_task_without_viewers_still_cleans_work_copy(self):
        response = self.start("import time; time.sleep(0.2); print('finished offline')")
        first = json.loads(next(iter(response.response)))
        current = server.ACTIVE_RUN
        self.assertIsNotNone(current)
        response.close()
        self.wait_closed(current)
        self.assertEqual(server.read_run(first["id"])["status"], "completed")
        self.assertFalse(self.work_path.exists())
        replay = self.events(self.get(f"/api/runs/{first['id']}/events"))
        self.assertEqual(replay[-1]["code"], 0)

    def test_nonzero_exit_keeps_complete_work_copy(self):
        record = b'{"role":"user","content":"saved before failure"}\n'
        response = self.start(
            "import os; f=open(os.environ['TEST_HISTORY'],'ab'); "
            f"f.write({record!r}); f.close(); raise SystemExit(7)"
        )
        events = self.events(response)
        response.close()
        self.assertEqual(events[-1]["code"], 7)
        self.assertEqual(server.read_run(events[0]["id"])["status"], "failed")
        self.assertEqual(self.work_path.read_bytes(), record)
        self.assertEqual((self.temp / "session.jsonl").read_bytes(), record)

    def test_zero_exit_with_partial_history_keeps_work_copy(self):
        complete = b'{"role":"user","content":"complete"}\n'
        partial = b'{"role":"assistant","content":"unfinished'
        response = self.start(
            "import os; f=open(os.environ['TEST_HISTORY'],'ab'); "
            f"f.write({complete + partial!r}); f.close()"
        )
        events = self.events(response)
        response.close()
        status = server.read_run(events[0]["id"])
        self.assertEqual(status["code"], 0)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["history_pending_bytes"], len(partial))
        self.assertEqual(self.work_path.read_bytes(), complete + partial)
        self.assertEqual((self.temp / "session.jsonl").read_bytes(), complete)

    def test_final_checkpoint_failure_keeps_copy_even_with_zero_exit(self):
        import threading
        real_commit = server.HistoryCheckpoint.commit
        def fail_final(checkpoint):
            if threading.current_thread().name.startswith("agent-"):
                raise OSError("simulated final checkpoint failure")
            return real_commit(checkpoint)
        with patch.object(server.HistoryCheckpoint, "commit", fail_final), \
                patch.object(server.app.logger, "exception"):
            response = self.start("print('normal process exit')")
            events = self.events(response)
            response.close()
        status = server.read_run(events[0]["id"])
        self.assertEqual(status["code"], 0)
        self.assertEqual(status["status"], "failed")
        self.assertTrue(self.work_path.is_file())
        self.assertIn("History 提交失败", status["error"])

    def test_collector_error_retains_copy(self):
        real_emit = server.AgentRun.emit
        def fail_output(current, kind, **fields):
            if kind == "output":
                raise OSError("simulated event write failure")
            return real_emit(current, kind, **fields)
        with patch.object(server.AgentRun, "emit", fail_output), \
                patch.object(server.app.logger, "exception"):
            response = self.start("print('output triggers failure')")
            events = self.events(response)
            response.close()
        self.assertEqual(server.read_run(events[0]["id"])["status"], "failed")
        self.assertTrue(self.work_path.is_file())

    def test_completion_metadata_failure_retains_copy(self):
        real_save = server.AgentRun.save_metadata
        def fail_completion(current):
            if current.metadata["status"] != "running":
                raise OSError("simulated completion metadata failure")
            return real_save(current)
        with patch.object(server.AgentRun, "save_metadata", fail_completion), \
                patch.object(server.app.logger, "exception"):
            response = self.start("print('normal exit')")
            self.events(response)
            response.close()
        self.assertTrue(self.work_path.is_file())
        self.assertFalse(server.RUN_LOCK.locked())

    def test_final_log_sync_failure_retains_copy(self):
        import os
        if os.name != "posix" or not Path("/proc/self/fd").is_dir():
            self.skipTest("Uses Linux file descriptor metadata")
        real_fsync = server.os.fsync
        def fail_log(fd):
            if os.readlink(f"/proc/self/fd/{fd}") == str(server.LOG_PATH):
                raise OSError("simulated log sync failure")
            return real_fsync(fd)
        with patch.object(server.os, "fsync", fail_log):
            response = self.start("print('normal exit')")
            events = self.events(response)
            response.close()
        status = server.read_run(events[0]["id"])
        self.assertEqual(status["status"], "failed")
        self.assertIn("日志落盘失败", status["error"])
        self.assertTrue(self.work_path.is_file())

    def test_cleanup_permission_failure_keeps_copy_and_task_success(self):
        real_unlink = Path.unlink
        def refuse_copy(path, *args, **kwargs):
            if path.name.endswith(".history.jsonl"):
                raise PermissionError("simulated cleanup denied")
            return real_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", refuse_copy), \
                patch.object(server.app.logger, "warning") as warning:
            response = self.start("print('completed')")
            events = self.events(response)
            response.close()
            self.assertTrue(warning.called)
        self.assertTrue(self.work_path.is_file())
        self.assertEqual(events[-1]["code"], 0)
        self.assertEqual(server.read_run(events[0]["id"])["status"], "completed")
        self.assertFalse(server.RUN_LOCK.locked())

    def test_cleanup_after_writers_stop_under_task_lock(self):
        real_discard = server.HistoryCheckpoint.discard_committed_work_copy
        observed = []
        def checked_discard(checkpoint):
            current = server.ACTIVE_RUN
            self.assertIsNotNone(current)
            self.assertTrue(server.RUN_LOCK.locked())
            self.assertTrue(current.lock.locked())
            self.assertEqual(current.process.poll(), 0)
            self.assertFalse(current.checkpoint_worker.is_alive())
            metadata = json.loads(server.run_path(current.id, "json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(checkpoint.pending_bytes, 0)
            observed.append(current.id)
            return real_discard(checkpoint)
        with patch.object(server.HistoryCheckpoint, "discard_committed_work_copy", checked_discard):
            response = self.start("print('ok')")
            events = self.events(response)
            response.close()
        self.assertEqual(observed, [events[0]["id"]])
        self.assertFalse(self.work_path.exists())

    def test_stop_before_collection_preserves_copy_even_if_exit_code_zero(self):
        # Simulate Stop racing with process exit, before final collection.
        real_collect = server.AgentRun.collect
        def stopped_collect(current):
            current.process.wait()
            current.stop()
            return real_collect(current)
        with patch.object(server.AgentRun, "collect", stopped_collect):
            response = self.start("pass")
            events = self.events(response)
            response.close()
        status = server.read_run(events[0]["id"])
        self.assertEqual(status["code"], 0)
        self.assertEqual(status["status"], "stopped")
        self.assertTrue(self.work_path.is_file())


if __name__ == "__main__":
    unittest.main()
