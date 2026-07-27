import argparse
import json
import os
from datetime import datetime
from pathlib import Path

try:
    from .backends import get_backend
    from .actions import ActionRunner
except ImportError:
    from backends import get_backend
    from actions import ActionRunner

base_dir = Path(__file__).resolve().parent
temp_dir = base_dir / "../temp"
temp_dir.mkdir(parents=True, exist_ok=True)
_env = os.environ.copy()
_env["TORCH_CPP_LOG_LEVEL"] = "ERROR"


class Agent:
    def __init__(
        self,
        max_tokens: int,
        max_steps: int,
        history: str,
    ):
        self.backend = get_backend()
        self.actions = ActionRunner()
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        if not history.endswith(".jsonl"):
            history += ".jsonl"
        self.history_file = Path(os.path.join(temp_dir, history)).expanduser()
        self.run_id = ""
        self.memory = []
        self.react_note = """IMPORTANT:
You are running in a ReAct loop:
<|thought|>...<\\thought|> -> <|action|>...<\\action|> ->
<|observation|>...<\\observation|> -> <|thought|>...<\\thought|>.
Every opening tag must have its matching closing tag.
When you need to use a tool, output exactly one action and then stop immediately.
Never output a second paired thought or action before the system returns an observation.
Never write an observation yourself; only the system writes it after an action runs.
Never repeat or quote a previous action block in your response.
After <\\action_input|>, output nothing else and wait for an observation.
Use this format:
<|thought|>explain the next step briefly.<\\thought|>
<|action|>one available action name<\\action|>
<|action_input|>
```text
arguments for the action
```
<\\action_input|>

After the system returns a paired observation, use it to decide the next step.
When the task is complete, do not call another action. Return:
<|thought|>brief summary of what you learned.<\\thought|>
<|final_answer|>the answer for the user.<\\final_answer|>

If an observation shows an action failed, reason about why it failed and try a corrected action.
For requests about current or latest information, verify the date and facts with actions.
Never present model memory, estimates, or unverified claims as current action results.
"""

    def load_history(self) -> list:
        if not self.history_file.is_file():
            return []

        history = []
        try:
            with self.history_file.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(
                            "==== SYSTEM ==== Ignoring invalid history line "
                            f"{line_number}: {exc}",
                            flush=True,
                        )
                        continue

                    if (
                        isinstance(item, dict)
                        and item.get("role") in {"user", "assistant"}
                        and isinstance(item.get("content"), str)
                    ):
                        history.append(
                            {
                                "role": item["role"],
                                "content": item["content"],
                            }
                        )
        except OSError as exc:
            print(f"==== SYSTEM ==== Failed to load history: {exc}", flush=True)
        return history

    def save_history(self, role: str, content: str, state: str) -> None:
        item = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "state": state,
            "role": role,
            "content": content,
        }
        try:
            with self.history_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            print(f"==== SYSTEM ==== Failed to save history: {exc}", flush=True)

    def run(self, text: str) -> None:
        now = datetime.now().astimezone()
        current_time = now.isoformat(timespec="seconds")
        self.run_id = now.strftime("%Y%m%dT%H%M%S%f%z")
        system_prompt = f"""You are an autonomous ReAct agent.

Current local date and time: {current_time}

Available actions:
{self.actions.desc()}

{self.react_note}
"""
        user_prompt = f"{text}"
        history = self.load_history()
        self.memory = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_prompt},
        ]
        self.save_history("user", user_prompt, "user")
        state = "model"
        steps = 0
        response = ""

        while state != "done":
            if state == "model":
                if steps >= self.max_steps:
                    state = "max_steps"
                    continue

                steps += 1
                response = ""
                printed_length = 0
                response_stream = self.backend.stream_response(
                    self.memory,
                    self.max_tokens,
                )
                try:
                    for chunk in response_stream:
                        response += chunk
                        action_end = self.actions.first_complete_action_end(response)
                        output_end = (
                            action_end if action_end is not None else len(response)
                        )
                        if output_end > printed_length:
                            print(
                                response[printed_length:output_end],
                                end="",
                                flush=True,
                            )
                            printed_length = output_end
                        if action_end is not None:
                            response = response[:action_end]
                            break
                finally:
                    close_stream = getattr(response_stream, "close", None)
                    if close_stream:
                        close_stream()

                self.memory.append({"role": "assistant", "content": response})
                self.save_history("assistant", response, "model")

                action_call = self.actions.parse_action(response)
                has_final_answer = bool(
                    self.actions.final_answer_pattern.search(response)
                )
                if has_final_answer and not action_call:
                    state = "done"
                else:
                    state = "action"

            elif state == "action":
                result = self.actions(response)
                if result is None:
                    state = "repair"
                else:
                    observation = f"<|observation|>{result}<\\observation|>"
                    print(f"\n\n{observation}\n", flush=True)
                    self.memory.append({"role": "user", "content": observation})
                    self.save_history("user", observation, "action")
                    state = "model"

            elif state == "repair":
                observation = (
                    "No valid action was found. Continue with either a valid "
                    "paired action or a paired final answer."
                )
                observation = f"<|observation|>{observation}<\\observation|>"
                print(f"\n\n{observation}\n", flush=True)
                self.memory.append({"role": "user", "content": observation})
                self.save_history("user", observation, "repair")
                state = "model"

            elif state == "max_steps":
                observation = (
                    f"Reached max_steps={self.max_steps}. Stop and provide the best final "
                    "answer from the collected observations."
                )
                observation = f"<|observation|>{observation}<\\observation|>"
                print(f"\n\n{observation}\n", flush=True)
                self.memory.append({"role": "user", "content": observation})
                self.save_history("user", observation, "max_steps")

                response = ""
                for chunk in self.backend.stream_response(self.memory, self.max_tokens):
                    response += chunk
                    print(chunk, end="", flush=True)
                self.memory.append({"role": "assistant", "content": response})
                self.save_history("assistant", response, "max_steps_final")
                state = "done"

        print("", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AToy: An agent create shit")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--input", type=str, default="None")
    parser.add_argument("--history", type=str, default="history")
    args = parser.parse_args()
    agent = Agent(
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        history=args.history,
    )
    with open(args.input.strip(), "r") as f:
        text = f.read()
    agent.run(text)
