import re
import inspect
import traceback
import subprocess
from dataclasses import dataclass


def python_code(code: str) -> str:
    """
    Directly write a Python script and use _result_ to represent the string result to be returned.
    """
    try:
        scope = {}
        exec(code, scope)
        return str(scope["_result_"])
    except Exception:
        return traceback.format_exc()


def run_cmd(cmd: str) -> str:
    """
    Directly write a BASH script (Linux). Return the string result.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"returncode:{e.returncode}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
    except Exception:
        return traceback.format_exc()


GLOBAL_ACTIONS = [
    python_code,
    run_cmd,
]


@dataclass
class ActionCall:
    name: str
    args: str


class ActionRunner:
    def __init__(self):
        self.react_pattern = re.compile(
            r"(?ims)^\s*Action\s*:\s*([A-Za-z_]\w*)\s*$\s*^Action\s*Input\s*:\s*([\s\S]*)"
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
                return output if output else "[action returned empty output]"
            else:
                return f"Invalid action name: {action_name}"
        return None

    def parse_action(self, text: str):
        react_match = self.react_pattern.search(text)
        if not react_match:
            return None

        action_input = react_match.group(2)
        action_input = re.split(
            r"(?im)^\s*(?:Observation|Final Answer)\s*:",
            action_input,
            maxsplit=1,
        )[0]
        return ActionCall(
            name=react_match.group(1).strip(),
            args=self._clean_action_input(action_input),
        )

    def _clean_action_input(self, text: str) -> str:
        text = text.strip()
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).rstrip()
        return text

    def desc(self) -> str:
        text_head = """Use the ReAct format when an action is required:
Thought: explain what you need to do next.
Action: $ACTION
Action Input:
```text
$ARGS
```

Below are the available $ACTION options along with their descriptions and code:\n\n"""
        text_actions = []
        for key, value in self.actions.items():
            text_actions.append(f"$ACTION={key}\n{value['desc']}")
        text_actions = "\n".join(text_actions)
        return text_head + text_actions


if __name__ == "__main__":
    action_runner = ActionRunner()
    print(action_runner.desc())
