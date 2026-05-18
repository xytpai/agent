import re
import inspect
import traceback
import subprocess


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


class ActionRunner:
    def __init__(self):
        self.pattern = r"\[\[\[(.*?)\]\]\]<<<<<([\s\S]*?)>>>>>"
        global GLOBAL_ACTIONS
        self.actions = {}
        for function in GLOBAL_ACTIONS:
            name = str(function.__name__)
            desc = str(inspect.getsource(function))
            self.actions[name] = {"name": name, "desc": desc, "func": function}

    def __call__(self, text: str):
        m = re.match(self.pattern, text)
        if m:
            action_name = m.group(1)
            action_args = m.group(2)
            return str(self.actions[action_name]["func"](action_args))
        return None

    def desc(self) -> str:
        text_head = """Output [[[$ACTION]]]<<<<<$ARGS>>>>> to indicate that an action is required.
Below are the available $ACTION options along with their descriptions and code:\n\n"""
        text_actions = []
        for key, value in self.actions.items():
            text_actions.append(f"$ACTION={key}\n{value['desc']}")
        text_actions = "\n".join(text_actions)
        return text_head + text_actions


if __name__ == "__main__":
    action_runner = ActionRunner()
    print(action_runner.desc())
