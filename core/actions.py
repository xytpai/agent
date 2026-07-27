import base64
import html
import inspect
import ipaddress
import json
import os
import re
import socket
import subprocess
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
import trafilatura
from ddgs import DDGS
from lxml import html as lxml_html


def search_web(request: str) -> str:
    """
    Search the live web and return normalized results with titles, URLs, and snippets.
    Input may be a plain query or JSON:
    {"query": "...", "max_results": 5, "region": "us-en", "timelimit": "d|w|m|y"}.
    The default region is automatically set to cn-zh for Chinese queries.
    """
    try:
        options = _parse_action_request(request, "query")
        query = str(options.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        if len(query) > 500:
            raise ValueError("query must be 500 characters or fewer")

        max_results = max(1, min(int(options.get("max_results", 5)), 10))
        default_region = "cn-zh" if re.search(r"[\u3400-\u9fff]", query) else "us-en"
        region = str(options.get("region", default_region)).strip() or default_region
        timelimit = options.get("timelimit")
        if timelimit not in {None, "d", "w", "m", "y"}:
            raise ValueError("timelimit must be one of: d, w, m, y")

        requested_backend = str(options.get("backend", "")).strip()
        raw_results = []
        backend_errors = []
        used_backend = ""

        if not requested_backend or requested_backend == "bing_html":
            try:
                raw_results = _search_bing_html(
                    query,
                    max_results,
                    region,
                    timelimit,
                )
                if raw_results:
                    used_backend = "bing_html"
            except Exception as exc:
                backend_errors.append(f"bing_html: {type(exc).__name__}")

        if not raw_results and requested_backend != "bing_html":
            backends = (
                [requested_backend]
                if requested_backend
                else [
                    "duckduckgo",
                    "brave",
                    "mojeek",
                    "startpage",
                    "google",
                    "yahoo",
                    "yandex",
                    "auto",
                ]
            )
            for backend in backends:
                try:
                    raw_results = DDGS().text(
                        query,
                        region=region,
                        safesearch="moderate",
                        timelimit=timelimit,
                        backend=backend,
                        max_results=max_results,
                    )
                    if raw_results:
                        used_backend = backend
                        break
                except Exception as exc:
                    backend_errors.append(f"{backend}: {type(exc).__name__}")
        if not raw_results:
            raise RuntimeError(
                "all search backends failed"
                + (f" ({', '.join(backend_errors)})" if backend_errors else "")
            )

        results = []
        for item in raw_results or []:
            url = item.get("href") or item.get("url") or ""
            if not url:
                continue
            results.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(url).strip(),
                    "snippet": str(item.get("body") or "").strip()[:1500],
                }
            )

        return json.dumps(
            {
                "query": query,
                "retrieved_at": _utc_now(),
                "backend": used_backend,
                "result_count": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return _action_error("search_web", exc)


def fetch_url(request: str) -> str:
    """
    Fetch a public HTTP(S) URL and extract readable page content.
    Input may be a plain URL or JSON: {"url": "...", "max_chars": 20000}.
    Private/local network addresses and binary responses are rejected.
    """
    try:
        options = _parse_action_request(request, "url")
        current_url = str(options.get("url", "")).strip()
        if not current_url:
            raise ValueError("url is required")
        max_chars = max(1000, min(int(options.get("max_chars", 20000)), 50000))
        max_bytes = 2 * 1024 * 1024
        response_url = current_url
        content_type = ""
        encoding = "utf-8"
        body = bytearray()
        download_truncated = False

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AToyAgent/1.0; " "+https://github.com/)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/json,"
            "text/plain,application/xml;q=0.9,*/*;q=0.1",
        }
        timeout = httpx.Timeout(20.0, connect=10.0)
        with httpx.Client(timeout=timeout, verify=True) as client:
            for redirect_count in range(6):
                _validate_public_url(response_url)
                with client.stream(
                    "GET",
                    response_url,
                    headers=headers,
                    follow_redirects=False,
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("redirect response has no Location header")
                        if redirect_count == 5:
                            raise ValueError("too many redirects")
                        response_url = urljoin(str(response.url), location)
                        continue

                    response.raise_for_status()
                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type and not (
                        content_type.startswith("text/")
                        or content_type
                        in {
                            "application/json",
                            "application/ld+json",
                            "application/xhtml+xml",
                            "application/xml",
                        }
                    ):
                        raise ValueError(f"unsupported content type: {content_type}")

                    encoding = response.encoding or "utf-8"
                    for chunk in response.iter_bytes():
                        remaining = max_bytes - len(body)
                        if remaining <= 0:
                            download_truncated = True
                            break
                        body.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            download_truncated = True
                            break
                    response_url = str(response.url)
                    break
            else:
                raise ValueError("unable to resolve URL")

        raw_text = bytes(body).decode(encoding, errors="replace")
        title = ""
        if content_type in {"application/json", "application/ld+json"}:
            try:
                content = json.dumps(
                    json.loads(raw_text),
                    ensure_ascii=False,
                    indent=2,
                )
            except json.JSONDecodeError:
                content = raw_text
        elif "html" in content_type or "<html" in raw_text[:1000].lower():
            content = trafilatura.extract(
                raw_text,
                url=response_url,
                output_format="markdown",
                include_links=True,
                include_tables=True,
                include_comments=False,
                favor_recall=True,
            )
            if not content:
                content = trafilatura.html2txt(raw_text)
            title_match = re.search(
                r"(?is)<title[^>]*>(.*?)</title>",
                raw_text,
            )
            if title_match:
                title = html.unescape(
                    re.sub(r"<[^>]+>", "", title_match.group(1))
                ).strip()
        else:
            content = raw_text

        content = (content or "").strip()
        content_truncated = len(content) > max_chars
        if content_truncated:
            content = content[:max_chars].rstrip()

        return json.dumps(
            {
                "url": response_url,
                "title": title,
                "retrieved_at": _utc_now(),
                "content_type": content_type or "unknown",
                "truncated": download_truncated or content_truncated,
                "content": content,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return _action_error("fetch_url", exc)


def _search_bing_html(
    query: str,
    max_results: int,
    region: str,
    timelimit: str | None,
) -> list:
    language = "zh-cn" if region.lower().startswith("cn-") else "en-us"
    params = {
        "q": query,
        "count": str(max(10, max_results * 3)),
        "setlang": language,
    }
    if language == "en-us":
        params["ensearch"] = "1"
    if timelimit in {"d", "w", "m"}:
        params["freshness"] = {"d": "Day", "w": "Week", "m": "Month"}[timelimit]

    response = httpx.get(
        "https://www.bing.com/search",
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    document = lxml_html.fromstring(response.text)
    results = []
    seen_urls = set()
    for node in document.xpath(
        "//li[contains(concat(' ', normalize-space(@class), ' '), " "' b_algo ')]"
    ):
        links = node.xpath(".//h2/a[1]")
        if not links:
            continue
        url = _decode_bing_url(str(links[0].get("href") or "").strip())
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = " ".join(part.strip() for part in links[0].itertext() if part.strip())
        snippets = node.xpath(
            ".//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' b_caption ')]//p[1]"
        )
        snippet = (
            " ".join(part.strip() for part in snippets[0].itertext() if part.strip())
            if snippets
            else ""
        )
        if not _search_result_is_relevant(query, title, url, snippet):
            continue
        results.append({"title": title, "href": url, "body": snippet})
        if len(results) >= max_results:
            break
    return results


def _decode_bing_url(url: str) -> str:
    parsed = urlparse(url)
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded.startswith("a1"):
        return url
    try:
        payload = encoded[2:]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
        return decoded if decoded.startswith(("http://", "https://")) else url
    except (ValueError, UnicodeDecodeError):
        return url


def _search_result_is_relevant(
    query: str,
    title: str,
    url: str,
    snippet: str,
) -> bool:
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]{2,}", query)
        if term.lower() not in {"and", "for", "from", "the", "with"}
    }
    terms.update(re.findall(r"[\u3400-\u9fff]{2,}", query))
    if not terms:
        return True

    haystack = f"{title} {url} {snippet}".lower()
    matches = sum(term.lower() in haystack for term in terms)
    return matches >= (1 if len(terms) <= 2 else 2)


def _parse_action_request(request: str, primary_key: str) -> dict:
    text = request.strip()
    if not text:
        return {}
    if not text.startswith("{"):
        return {primary_key: text}

    options = json.loads(text)
    if not isinstance(options, dict):
        raise ValueError("JSON action input must be an object")
    return options


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("only http and https URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("credentials in URLs are not allowed")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("local network URLs are not allowed")

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("private or non-public network URLs are not allowed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _action_error(action: str, exc: Exception) -> str:
    return json.dumps(
        {
            "action": action,
            "error": f"{type(exc).__name__}: {exc}",
            "retrieved_at": _utc_now(),
        },
        ensure_ascii=False,
        indent=2,
    )


def python_code(code: str) -> str:
    """
    Directly write a Python script and use _result_ to represent the string result to be returned.
    """
    try:
        scope = {}
        exec(code, scope)
        if "_result_" not in scope:
            return "[python_code completed without setting _result_]"
        return _format_text_result(str(scope["_result_"]))
    except Exception:
        return traceback.format_exc()


def run_cmd(cmd: str) -> str:
    """
    Directly write a shell script. Return the command status, stdout, and stderr.
    """
    try:
        kwargs, shell_error = _shell_kwargs_for(cmd)
        if shell_error:
            return shell_error

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            **kwargs,
        )
        return _format_command_result(result.returncode, result.stdout, result.stderr)
    except Exception:
        return traceback.format_exc()


def _shell_kwargs_for(cmd: str):
    bash_path = shutil.which("bash")
    if os.name != "nt":
        return ({"executable": bash_path} if bash_path else {}), None

    if not _uses_posix_shell_syntax(cmd):
        return {}, None

    if bash_path and _bash_works(bash_path):
        return {"executable": bash_path}, None

    return {}, "\n".join(
        [
            "command not executed",
            "reason: POSIX shell syntax was detected, but no usable bash was found on Windows.",
            "hint: use python_code for Python snippets, or rewrite the command for Windows shell.",
        ]
    )


def _uses_posix_shell_syntax(cmd: str) -> bool:
    return bool(re.search(r"<<\s*['\"]?\w+|/dev/null|\|\|\s*true", cmd))


def _bash_works(bash_path: str) -> bool:
    try:
        result = subprocess.run(
            [bash_path, "-lc", "true"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _format_text_result(text: str) -> str:
    return text if text.strip() else "[action completed with empty output]"


def _format_stream(name: str, text: str) -> str:
    text = text.replace("\x00", "").rstrip()
    return f"{name}:\n{text if text else '[empty]'}"


def _format_command_result(returncode: int, stdout: str, stderr: str) -> str:
    status = "succeeded" if returncode == 0 else "failed"
    return "\n".join(
        [
            f"command {status}",
            f"returncode: {returncode}",
            _format_stream("stdout", stdout),
            _format_stream("stderr", stderr),
        ]
    )


GLOBAL_ACTIONS = [
    search_web,
    fetch_url,
    python_code,
    run_cmd,
]


@dataclass
class ActionCall:
    name: str
    args: str
    warning: str = ""


class ActionRunner:
    def __init__(self):
        self.action_pattern = re.compile(
            r"(?is:<\|action\|>\s*([A-Za-z_]\w*)\s*<\\action\|>)"
            r"|(?i:Action[ \t]*:[ \t]*([A-Za-z_]\w*))"
        )
        self.action_input_pattern = re.compile(
            r"(?is:<\|action_input\|>(.*?)<\\action_input\|>)"
            r"|(?i:Action[ \t]*Input[ \t]*:[ \t]*)"
        )
        self.final_answer_pattern = re.compile(
            r"(?is:<\|final_answer\|>.*?<\\final_answer\|>)"
            r"|(?i:Final Answer[ \t]*:)"
        )
        self.block_start_pattern = re.compile(
            r"(?i:<\|(?:thought|action|observation|final_answer)\|>)"
            r"|(?i:(?:Thought|Action|Observation|Final Answer)[ \t]*:)"
        )
        global GLOBAL_ACTIONS
        self.actions = {}
        for function in GLOBAL_ACTIONS:
            name = str(function.__name__)
            desc = (
                f"{name}{inspect.signature(function)}\n"
                f"{inspect.getdoc(function) or 'No description available.'}"
            )
            self.actions[name] = {"name": name, "desc": desc, "func": function}

    def __call__(self, text: str):
        action_call = self.parse_action(text)
        if action_call:
            action_name = action_call.name
            action_args = action_call.args
            if self.actions.get(action_name, None):
                output = str(self.actions[action_name]["func"](action_args))
                output = _format_text_result(output)
                if action_call.warning:
                    return f"{action_call.warning}\n{output}"
                return output
            else:
                return f"Invalid action name: {action_name}"
        return None

    def first_complete_action_end(self, text: str):
        action_match = self.action_pattern.search(text)
        if not action_match or action_match.group(1) is None:
            return None

        tail = text[action_match.end() :]
        action_input_match = self.action_input_pattern.search(tail)
        if not action_input_match or action_input_match.group(1) is None:
            return None

        next_action_match = self.action_pattern.search(tail)
        if next_action_match and next_action_match.start() < action_input_match.start():
            return None

        return action_match.end() + action_input_match.end()

    def parse_action(self, text: str):
        action_matches = list(self.action_pattern.finditer(text))
        if not action_matches:
            return None

        action_match = action_matches[0]
        tail = text[action_match.end() :]
        action_input_match = self.action_input_pattern.search(tail)
        next_action_match = self.action_pattern.search(tail)
        if action_match.group(1) is not None and (
            not action_input_match
            or action_input_match.group(1) is None
            or (
                next_action_match
                and next_action_match.start() < action_input_match.start()
            )
        ):
            return None

        if not action_input_match or (
            next_action_match and next_action_match.start() < action_input_match.start()
        ):
            return ActionCall(
                name=self._action_name(action_match),
                args="",
                warning=(
                    "[warning: paired <|action_input|> was missing; "
                    "executed with empty input]"
                ),
            )

        action_input = (
            action_input_match.group(1)
            if action_input_match.group(1) is not None
            else tail[action_input_match.end() :]
        )
        warning = ""
        if len(action_matches) > 1:
            warning = (
                f"[warning: response contained {len(action_matches)} "
                "<|action|> blocks; executed only the first one]"
            )

        return ActionCall(
            name=self._action_name(action_match),
            args=self._extract_action_input(action_input),
            warning=warning,
        )

    @staticmethod
    def _action_name(action_match: re.Match) -> str:
        return next(
            group.strip() for group in action_match.groups() if group is not None
        )

    def _extract_action_input(self, text: str) -> str:
        text = text.lstrip()
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            for idx, line in enumerate(lines[1:], start=1):
                if line.strip() == "```":
                    return "\n".join(lines[1:idx]).rstrip()

        action_input = self.block_start_pattern.split(text, maxsplit=1)[0]
        return action_input.strip()

    def desc(self) -> str:
        shell_note = (
            "On Windows, run_cmd uses the default Windows shell. Do not use POSIX "
            "heredocs such as `python - <<'PY'`; use python_code for Python snippets "
            "or write Windows-compatible commands."
            if os.name == "nt"
            else "On POSIX systems, run_cmd uses bash when it is available."
        )
        text_head = f"""Use the tagged ReAct format when an action is required:
<|thought|>explain what you need to do next.<\\thought|>
<|action|>$ACTION<\\action|>
<|action_input|>
```text
$ARGS
```
<\\action_input|>

Every opening tag must have its matching closing tag.
Output exactly one paired <|action|> block, then stop and wait for the paired <|observation|> block.
{shell_note}

For current or factual web research, use search_web to discover sources, then
use fetch_url to read the most relevant pages. Search snippets are discovery
hints, not sufficient evidence. Prefer primary sources, preserve source URLs
and publication dates, and never replace a requested metric with a proxy
without clearly explaining the substitution. If results are empty or
irrelevant, retry with more specific terms or an English-language query for
international sources.

Below are the available action names and their descriptions and code:\n\n"""
        text_actions = []
        for key, value in self.actions.items():
            text_actions.append(f"<|action|>{key}<\\action|>\n{value['desc']}")
        text_actions = "\n".join(text_actions)
        return text_head + text_actions


if __name__ == "__main__":
    action_runner = ActionRunner()
    print(action_runner.desc())
