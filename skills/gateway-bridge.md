---
name: gateway-bridge
description: Build a temporary bridge using an HTTPS forwarding service and an SSH reverse tunnel when a GPU or remote machine cannot reach an OpenAI-compatible gateway but the agent host can. Use for remote GPU hosts, ProxyJump, internal APIs, SSE streaming, and connection timeout troubleshooting.
---

# Gateway Bridge

## Instructions for the AI Executing This Skill

This is a complete, single-file skill. **No companion script files are required.**
Read the operating rules and prerequisites first, then choose the appropriate task:
explain, inspect, start, or stop. Do not start services when the user only requests
an explanation or documentation changes.

When the user authorizes an actual bridge:

1. Confirm the agent execution host, target SSH alias and jump host, upstream URL,
   remote port, and authorization.
2. Inspect existing bridges first. The scripts below do not take over bridges
   created manually.
3. Create a private working directory with mode 700 on the agent host, for example:
   `WORK=$(mktemp -d "${TMPDIR:-/tmp}/gateway-bridge-work.XXXXXXXX")`.
4. Create `$WORK/scripts` and use a file-writing tool to save the code in
   Appendices A and B verbatim as `$WORK/scripts/proxy.py` and
   `$WORK/scripts/bridge.py`. These are runtime artifacts generated from this
   document, not external skill dependencies. Keep both files in the same
   directory because `bridge.py` imports `proxy.py`.
5. For offline testing, also create `$WORK/tests` and save Appendix C as
   `$WORK/tests/test_proxy.py`.
6. **Run every relative-path command below from `$WORK`, not from the directory
   containing SKILL.md.** Invoke the scripts with `python3`; executable
   permissions are not required.
7. Do not store private keys or API keys in the working directory. Keep both the
   working directory and state directory while the bridge is in use. Report
   their paths, the process host, the remote BASE_URL, and verification results.
8. Use the generated `bridge.py` for subsequent management. If the working
   directory is lost, regenerate it from this document, but use the original
   `--name` and `--state-root`. Do not blindly start another bridge.
9. Remove the generated working directory only after confirming the bridge has
   stopped. Do not delete user identity files or terminate unidentified processes.

Credentials must come from securely provisioned user files or environment
variables. Never include them in this document or tool output.
Example hosts and model names are not evidence about the current environment.

All hostnames under `example.invalid`, SSH aliases, bridge names, ports, and
gateway paths in this document are illustrative. Replace them with user-approved
values before execution. In particular, provide the actual HTTPS gateway URL
with `--upstream`; the placeholder default is not a working service. The example
`/OpenAI` path is not universal: use the upstream's actual base path consistently
in connectivity checks and the remote application's BASE_URL.

## Architecture and Scope

~~~text
Remote application -> 127.0.0.1:18080 on the remote host
                   -> SSH reverse tunnel, optionally through a jump host
                   -> 127.0.0.1:<automatic-port> on the agent host
                   -> HTTPS gateway /OpenAI/...
~~~

"Local" means **the host executing the agent tool**, not necessarily the user's
computer. The background Python forwarder, SSH client process, and jump-host
connection run on the agent host. No Python service is deployed on the remote host; its SSH
server provides the reverse-forwarding listener.

This is a temporary development bridge, not a production gateway. It has no
systemd integration, boot-time startup, or automatic reconnection. It depends on
the agent host and background processes remaining alive.

Listeners are restricted to loopback by default, but other users on the same
machine may still access them. This is not a multi-tenant security boundary.
API keys travel over HTTP on loopback, through SSH between machines, and over
certificate-verified HTTPS to the upstream gateway.

## Agent Operating Rules

1. Explain that background processes will run on the **agent host** and a loopback
   listener will be opened on the target host. Confirm user authorization.
2. Check hostname, direct gateway reachability, SSH routing, ports, and existing
   bridges before starting anything.
3. Do not ask users to paste private keys or API keys into chat. Use a securely
   provisioned identity file or SSH agent.
4. Recommend revoking or rotating exposed credentials. Never place them in the
   skill, repository, examples, or logs.
5. Verify jump-host and target-host SSH fingerprints through a trusted channel.
   The scripts require established host trust and do not disable host-key checks.
6. Test without an API key first. HTTP 401, 403, or 404 indicates that an HTTP
   service responded, not that authentication or model inference succeeded.
7. If a tool call times out, inspect status before trying another start. Stop only
   processes managed by this skill.
8. Documenting this skill does not authorize taking over an existing bridge.
   Do not restart, migrate, or copy an existing bridge's private key without authorization.

## Prerequisites and Limitations

- The manager requires Linux `/proc`, Python 3.9+, and OpenSSH. The remote SSH
  server must permit reverse port forwarding.
- Python connects directly to the upstream and does not read HTTP_PROXY or
  HTTPS_PROXY. If curl works only through a corporate proxy, this implementation
  needs adaptation.
- TLS certificate verification remains enabled. Configure an appropriate corporate
  CA using SSL_CERT_FILE when needed; do not disable verification.
- Streaming responses, including SSE, are forwarded incrementally. Request bodies
  require Content-Length. Chunked uploads and WebSockets are not supported.
- Request bodies are limited to 128 MiB. Connection/read/write timeouts are
  600 seconds; this does not guarantee inference will complete.
- The scripts do not log request headers, request bodies, or API keys. SSH logs
  may contain usernames and hostnames; redact them before sharing.
- Setting BASE_URL in a shell does not guarantee that an application reads it.
  Check the client's actual configuration.

Run all `scripts/...` commands below from the `$WORK` directory generated from
the appendices.

## 1. Inspect the Agent Environment

~~~bash
hostname
command -v python3
command -v ssh
curl --noproxy '*' -sS -o /dev/null \
  --connect-timeout 5 --max-time 10 \
  -w 'HTTP=%{http_code} peer=%{remote_ip}\n' \
  https://gateway.example.invalid/OpenAI/models
~~~

Test the same direct network path that Python uses. A successful curl request
through an environment-configured proxy does not establish direct reachability.

## 2. Prepare SSH Without Overwriting Existing Configuration

Example `~/.ssh/config` on the agent host:

~~~sshconfig
Host bridge-jump
    HostName jump.example.invalid
    User <your-username>
    IdentityFile ~/.ssh/<securely-provisioned-private-key-file>
    IdentitiesOnly yes
    ForwardAgent no

Host bridge-target
    HostName target.example.invalid
    User <your-username>
    IdentityFile ~/.ssh/<securely-provisioned-private-key-file>
    IdentitiesOnly yes
    ProxyJump bridge-jump
    ForwardAgent no
~~~

Reuse an existing alias such as `bridge-target` when available. An encrypted private
key can be loaded securely into an SSH agent beforehand. The scripts do not copy
private keys or prompt for passwords. Identity files should normally have mode 600.

After verifying and recording both host fingerprints, test:

~~~bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes bridge-target hostname
~~~

Failure to resolve the target hostname on the agent host is not necessarily a
problem: ProxyJump can let the jump host resolve the target.
For a separate configuration file, use `ssh -F /absolute/path/config ...`.
The manager supports a corresponding option.

## 3. Start on the Agent Host

~~~bash
python3 scripts/bridge.py start \
  --name remote-bridge \
  --target bridge-target \
  --upstream https://gateway.example.invalid/OpenAI \
  --remote-port 18080
~~~

For a separate SSH configuration, append `--ssh-config /absolute/path/config`.

The background processes are conceptually equivalent to:

~~~text
python3 proxy.py --upstream https://gateway.example.invalid/OpenAI ...
ssh -M -S <control> -R 127.0.0.1:18080:127.0.0.1:<proxy-port> -NT <target>
~~~

SSH uses ExitOnForwardFailure, keepalives, and a dedicated control socket.
The default state directory is `~/.local/state/gateway-bridge/<name>/`, with
mode 700. It contains `state.json`, `ready.json`, `control`, `proxy.log`, and
`ssh.log`, but no private keys or API keys.

Override it with `--state-root /short/absolute/path` if needed; SSH control-socket
paths have length limits. Do not run concurrent management commands with the same
name. If a port is occupied, choose another port rather than terminating an
unidentified process.

## 4. Verify from the Remote Host Without an API Key First

~~~bash
ssh bridge-target '
  hostname
  ss -ltn "sport = :18080"
  curl --noproxy "*" -sS -o /dev/null \
    --connect-timeout 3 --max-time 10 \
    -w "HTTP=%{http_code} connect=%{time_connect}s first_byte=%{time_starttransfer}s total=%{time_total}s\n" \
    http://127.0.0.1:18080/OpenAI/models
'
~~~

If using a separate configuration, also add `-F /absolute/path/config`.

**Verify that the listener binds to 127.0.0.1, not 0.0.0.0 or [::].**
A remote sshd configured with `GatewayPorts yes` may force wildcard binding. If
that happens, stop the bridge immediately and ask the administrator to use
`GatewayPorts no` or an appropriate configuration allowing the client to select
the bind address before retrying.

Interpret results carefully:

- **401:** The HTTP path is reachable; authentication is missing or invalid.
- **403:** Permissions or access policy may be blocking the request; investigate.
- **404:** The path may be incorrect. A 404 at the gateway root does not mean
  network connectivity failed.
- **502:** The forwarder could not complete the upstream request. Check agent DNS,
  direct connectivity, and TLS/CA configuration.
- **Connection refused:** No listener, an incorrect port, or a container-network
  mismatch.
- **Timeout:** Check manager status, the remote listener, agent-to-gateway direct
  access, and then the remote request. Use short timeouts to isolate the issue.

Report successful inference only after a real model request succeeds using a key
securely configured by the user. A user-supplied MODEL_NAME does not establish
model availability or authorization.

## 5. Configure the Remote Application

In the same terminal or job environment that launches the application:

~~~bash
export BASE_URL=http://127.0.0.1:18080/OpenAI
# The user must securely set API_KEY in the environment, not paste it into chat.
export MODEL_NAME='<an-authorized-model>'
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

# Only if the client uses these variable names:
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="$API_KEY"
~~~

Explicit configuration for the OpenAI Python SDK:

~~~python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["API_KEY"],
)
~~~

Paths are preserved: local `/OpenAI` maps to upstream `/OpenAI`. Do not append
`/v1` without confirming the gateway's required path.

Inside a container, 127.0.0.1 may refer to the container rather than the host.
Follow platform networking requirements, such as using host networking when
authorized. Do not expose the proxy publicly merely for convenience.

## 6. Inspect and Stop on the Agent Host

~~~bash
python3 scripts/bridge.py status --name remote-bridge
python3 scripts/bridge.py stop --name remote-bridge
~~~

If startup used `--state-root`, supply the same value here.

`status` checks managed processes and the SSH control connection. It does not
call the gateway or verify a model.

`stop` closes the SSH master and stops this skill's Python process. It retains
credential-free state and logs. Delete the associated state directory manually
only after confirming shutdown.

The manager records Linux process start times to avoid mistaking a reused PID
for the original process.

Existing manually created bridges, such as those under `/tmp/gateway-bridge-*`,
are not automatically managed by this skill. Before starting a new bridge,
either obtain authorization to stop the old one or choose a different port.
Never copy an old bridge's identity file into the repository.

## Offline Tests

~~~bash
python3 -m unittest discover -s tests -v
~~~

First generate `$WORK/tests/test_proxy.py` from Appendix C.
The tests use loopback HTTP and a mocked upstream only. They do not contact the
real gateway, use SSH, or access real credentials.


## Appendix A: HTTPS Forwarder Source

At runtime, save this code as `$WORK/scripts/proxy.py`:

~~~~python
#!/usr/bin/env python3
"""Loopback HTTP to a fixed HTTPS upstream; no request logging."""
import argparse
import http.client
import http.server
import json
import os
from pathlib import Path
import ssl
from urllib.parse import urlsplit

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailer", "transfer-encoding", "upgrade"}


def parse_upstream(url):
    p = urlsplit(url)
    if (p.scheme != "https" or not p.hostname or p.username is not None
            or p.password is not None or p.query or p.fragment):
        raise ValueError("upstream must be HTTPS without credentials/query/fragment")
    return p.hostname, p.port or 443, p.path.rstrip("/")


def handler_for(host, port, prefix, tls):
    class Proxy(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def forward(self):
            self.close_connection = True
            self.connection.settimeout(600)
            path = urlsplit(self.path)
            if (not self.path.startswith("/") or self.path.startswith("//")
                    or path.scheme or path.netloc
                    or not (not prefix or path.path == prefix
                            or path.path.startswith(prefix + "/"))):
                self.send_error(404)
                return
            if self.headers.get("Transfer-Encoding"):
                self.send_error(501, "Use Content-Length")
                return
            lengths = self.headers.get_all("Content-Length", [])
            try:
                if len(lengths) > 1:
                    raise ValueError()
                size = int(lengths[0]) if lengths else 0
                if size < 0:
                    raise ValueError()
            except ValueError:
                self.send_error(400)
                return
            if size > 128 * 1024 * 1024:
                self.send_error(413)
                return
            upstream = None
            started = False
            try:
                body = self.rfile.read(size) if size else None
                if body is not None and len(body) != size:
                    self.send_error(400)
                    return
                excluded = HOP | {"host", "expect"} | {
                    v.strip().lower()
                    for v in self.headers.get("Connection", "").split(",")
                }
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in excluded}
                headers["Connection"] = "close"
                upstream = http.client.HTTPSConnection(
                    host, port, context=tls, timeout=600)
                upstream.request(self.command, self.path, body=body, headers=headers)
                response = upstream.getresponse()
                self.send_response_only(response.status, response.reason)
                excluded = HOP | {
                    v.strip().lower()
                    for v in (response.getheader("Connection") or "").split(",")
                }
                for k, v in response.getheaders():
                    if k.lower() not in excluded:
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                if self.command != "HEAD":
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except Exception:
                if not started:
                    try:
                        self.send_error(502, "Gateway connection failed")
                    except Exception:
                        pass
            finally:
                if upstream:
                    upstream.close()

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

    return Proxy


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    host, port, prefix = parse_upstream(args.upstream)
    server = Server(("127.0.0.1", 0),
                    handler_for(host, port, prefix, ssl.create_default_context()))
    tmp = args.ready_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"port": server.server_port}))
    tmp.replace(args.ready_file)
    server.serve_forever()


if __name__ == "__main__":
    main()
~~~~


## Appendix B: Start, Status, and Stop Manager Source

At runtime, save this code as `$WORK/scripts/bridge.py`:

~~~~python
#!/usr/bin/env python3
"""Manage a temporary loopback gateway bridge on Linux."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from proxy import parse_upstream

HERE = Path(__file__).resolve().parent


def stamp(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, ValueError, IndexError):
        return None


def alive(record):
    return bool(record and record.get("stamp")
                and stamp(record["pid"]) == record["stamp"])


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def launch(command, logfile):
    with logfile.open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    return {"pid": process.pid, "stamp": stamp(process.pid)}


def ssh_base(config):
    command = ["ssh"]
    if config:
        command += ["-F", config]
    return command + ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                      "-o", "ForwardAgent=no", "-o", "ConnectTimeout=8",
                      "-o", "ConnectionAttempts=1"]


def control(data, operation):
    try:
        return subprocess.run(
            ssh_base(data.get("ssh_config")) +
            ["-S", data["control"], "-O", operation, data["target"]],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=4,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def stop(data):
    if alive(data.get("ssh")):
        control(data, "exit")
    for key in ("ssh", "proxy"):
        record = data.get(key)
        if alive(record):
            try:
                os.kill(record["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and any(
            alive(data.get(k)) for k in ("ssh", "proxy")):
        time.sleep(0.1)
    if any(alive(data.get(k)) for k in ("ssh", "proxy")):
        raise RuntimeError("A managed process is still running; inspect before cleanup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop"))
    parser.add_argument("--name", default="default")
    parser.add_argument("--state-root", type=Path,
                        default=Path.home() / ".local/state/gateway-bridge")
    parser.add_argument("--target", help="SSH alias or user@host")
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--upstream", default="https://gateway.example.invalid/OpenAI")
    parser.add_argument("--remote-port", type=int, default=18080)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Process management requires Linux /proc")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.name):
        parser.error("name must be 1-40 letters/digits/underscores/hyphens")
    os.umask(0o077)
    directory = args.state_root.expanduser().resolve() / args.name
    if directory.is_symlink():
        parser.error("state directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.stat().st_uid != os.getuid():
        parser.error("state directory is not owned by the current user")
    directory.chmod(0o700)
    statefile = directory / "state.json"
    data = json.loads(statefile.read_text()) if statefile.exists() else {}

    if args.command == "status":
        if not data:
            print("No managed bridge state:", directory)
            return
        print("State directory:", directory)
        for key in ("proxy", "ssh"):
            print(f"{key}: {'running' if alive(data.get(key)) else 'not running'}",
                  f"pid={data.get(key, {}).get('pid', '-')}")
        print("SSH control:", "OK" if alive(data.get("ssh")) and control(data, "check")
              else "unavailable")
        print("Remote BASE_URL:", data.get("base_url", "startup incomplete"))
        print("This is process/tunnel status, NOT a gateway or inference test.")
        return

    if args.command == "stop":
        if data:
            stop(data)
            data["stopped"] = True
            save(statefile, data)
        print("Managed bridge stopped (if present). Logs/state retained:", directory)
        return

    if not args.target or args.target.startswith("-") or any(
            c.isspace() for c in args.target):
        parser.error("start needs a non-option SSH target with no whitespace")
    if not 1024 <= args.remote_port <= 65535:
        parser.error("remote-port must be 1024..65535")
    _, _, prefix = parse_upstream(args.upstream)
    if any(alive(data.get(k)) for k in ("ssh", "proxy")):
        parser.error("A managed process already exists; use status/stop first")
    config = str(args.ssh_config.expanduser().resolve()) if args.ssh_config else None
    if config and not Path(config).is_file():
        parser.error("SSH config does not exist")
    socket = directory / "control"
    if len(os.fsencode(socket)) > 100:
        parser.error("ControlPath too long; choose a shorter --state-root")
    socket.unlink(missing_ok=True)
    ready = directory / "ready.json"
    ready.unlink(missing_ok=True)
    data = {"target": args.target, "ssh_config": config, "control": str(socket),
            "upstream": args.upstream,
            "base_url": f"http://127.0.0.1:{args.remote_port}{prefix}"}
    save(statefile, data)
    try:
        data["proxy"] = launch(
            [sys.executable, str(HERE / "proxy.py"), "--upstream", args.upstream,
             "--ready-file", str(ready)], directory / "proxy.log")
        save(statefile, data)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if not alive(data["proxy"]) or time.monotonic() > deadline:
                raise RuntimeError("Proxy startup failed; inspect proxy.log")
            time.sleep(0.1)
        local_port = json.loads(ready.read_text())["port"]
        data["local_port"] = local_port
        command = ssh_base(config) + [
            "-M", "-S", str(socket), "-o", "ControlPersist=no",
            "-o", "ForkAfterAuthentication=no",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-R", f"127.0.0.1:{args.remote_port}:127.0.0.1:{local_port}",
            "-NT", args.target]
        data["ssh"] = launch(command, directory / "ssh.log")
        save(statefile, data)
        deadline = time.monotonic() + 25
        while True:
            if not alive(data["ssh"]):
                raise RuntimeError("SSH startup failed; inspect ssh.log")
            if socket.exists() and control(data, "check"):
                time.sleep(0.3)
                if alive(data["ssh"]):
                    break
            if time.monotonic() > deadline:
                raise RuntimeError("SSH startup timed out; inspect ssh.log")
            time.sleep(0.2)
        print("Bridge processes started on:", os.uname().nodename)
        print("Remote BASE_URL:", data["base_url"])
        print("State directory:", directory)
        print("Next: verify remote loopback-only listener and no-key HTTP request.")
    except BaseException:
        stop(data)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
~~~~


## Appendix C: Optional Offline Test Source

At runtime, save this code as `$WORK/tests/test_proxy.py`:

~~~~python
"""Offline tests using loopback HTTP and a mocked HTTPS upstream."""
import http.client
import io
import os
from pathlib import Path
import ssl
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bridge
import proxy


class Response:
    status = 200
    reason = "OK"

    def __init__(self):
        self.body = io.BytesIO(b"data: hello\n\ndata: [DONE]\n\n")

    def getheader(self, name):
        return "keep-alive" if name.lower() == "connection" else None

    def getheaders(self):
        return [("Content-Type", "text/event-stream"),
                ("Transfer-Encoding", "chunked"), ("Connection", "keep-alive")]

    def read1(self, size):
        return self.body.read(min(size, 5))


class Upstream:
    calls = []

    def __init__(self, host, port, **kwargs):
        self.calls.append(("connect", host, port, kwargs))

    def request(self, method, path, body=None, headers=None):
        self.calls.append(("request", method, path, body, headers))

    def getresponse(self):
        return Response()

    def close(self):
        pass


class ProxyTest(unittest.TestCase):
    def setUp(self):
        Upstream.calls = []
        self.server = proxy.Server(
            ("127.0.0.1", 0),
            proxy.handler_for("example.invalid", 443, "/OpenAI",
                              ssl.create_default_context()))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    @patch.object(proxy.http.client, "HTTPSConnection", Upstream)
    def test_stream_and_forwarding(self):
        code, headers, body = self.request(
            "POST", "/OpenAI/chat/completions?test=1", b'{"stream":true}',
            {"Authorization": "Bearer offline-test-value",
             "Content-Type": "application/json"})
        self.assertEqual(code, 200)
        self.assertEqual(body, b"data: hello\n\ndata: [DONE]\n\n")
        self.assertEqual(headers["Connection"], "close")
        self.assertNotIn("Transfer-Encoding", headers)
        self.assertEqual(Upstream.calls[0][1:3], ("example.invalid", 443))
        self.assertEqual(Upstream.calls[1][1:4],
                         ("POST", "/OpenAI/chat/completions?test=1", b'{"stream":true}'))
        self.assertEqual(Upstream.calls[1][4]["Authorization"],
                         "Bearer offline-test-value")
        self.assertNotIn("Host", Upstream.calls[1][4])

    @patch.object(proxy.http.client, "HTTPSConnection", Upstream)
    def test_head(self):
        code, _, body = self.request("HEAD", "/OpenAI/models")
        self.assertEqual((code, body), (200, b""))

    @patch.object(proxy.http.client, "HTTPSConnection", Upstream)
    def test_path_rejected_without_upstream(self):
        self.assertEqual(self.request("GET", "/unrelated")[0], 404)
        self.assertEqual(self.request("GET", "http://example.invalid/OpenAI")[0], 404)
        self.assertEqual(Upstream.calls, [])

    def test_chunked_rejected(self):
        self.assertEqual(self.request(
            "POST", "/OpenAI/test", headers={"Transfer-Encoding": "chunked"})[0], 501)

    @patch.object(proxy.http.client, "HTTPSConnection", side_effect=OSError("offline"))
    def test_failure_returns_502(self, _):
        self.assertEqual(self.request("GET", "/OpenAI/models")[0], 502)

    def test_upstream_validation(self):
        for url in ("http://example.invalid/OpenAI",
                    "https://user:secret@example.invalid/OpenAI",
                    "https://example.invalid/OpenAI?api_key=test"):
            with self.assertRaises(ValueError):
                proxy.parse_upstream(url)
        self.assertEqual(proxy.parse_upstream("https://example.invalid/OpenAI/"),
                         ("example.invalid", 443, "/OpenAI"))

    def test_pid_identity(self):
        record = {"pid": os.getpid(), "stamp": bridge.stamp(os.getpid())}
        self.assertTrue(bridge.alive(record))
        record["stamp"] = "not-the-start-time"
        self.assertFalse(bridge.alive(record))


if __name__ == "__main__":
    unittest.main()
~~~~
