import re
import os
import inspect
import traceback
import subprocess
import shutil
from dataclasses import dataclass


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
            desc = str(inspect.getsource(function))
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

Below are the available action names and their descriptions and code:\n\n"""
        text_actions = []
        for key, value in self.actions.items():
            text_actions.append(f"<|action|>{key}<\\action|>\n{value['desc']}")
        text_actions = "\n".join(text_actions)
        return text_head + text_actions


if __name__ == "__main__":
    action_runner = ActionRunner()
    print(action_runner.desc())
