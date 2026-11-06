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

DEFAULT_TARGET_HISTORY_CONTEXT_SIZE = 2500
HISTORY_CUTOFF_STATE = "history_cutoff"
HISTORY_CUTOFF_NOTICE = (
    "[History cutoff] Earlier history is not loaded because it exceeded the "
    "context target; the unloaded history is in the JSONL file {history_file}, "
    "{line_range}; inspect only the needed lines with tools to recover the "
    "task and progress."
)


class Agent:
    def __init__(
        self,
        max_tokens: int,
        max_steps: int,
        history: str,
        effort: str | None = None,
        target_history_context_size: int = DEFAULT_TARGET_HISTORY_CONTEXT_SIZE,
    ):
        if target_history_context_size < 0:
            raise ValueError("target_history_context_size must be >= 0")
        self.target_history_context_size = target_history_context_size
        self.backend = get_backend(effort=effort)
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
<|think|>...<\\think|> -> <|action|>...<\\action|> ->
<|observation|>...<\\observation|> -> <|think|>...<\\think|>.
Every opening tag must have its matching closing tag.
When you need to use a tool, output exactly one action and then stop immediately.
Never output a second paired think block or action before the system returns an observation.
Never write an observation yourself; only the system writes it after an action runs.
Never repeat or quote a previous action block in your response.
After <\\action_input|>, output nothing else and wait for an observation.
Use this format:
<|think|>explain the next step briefly.<\\think|>
<|action|>one available action name<\\action|>
<|action_input|>
```text
arguments for the action
```
<\\action_input|>

After the system returns a paired observation, use it to decide the next step.
When the task is complete, do not call another action. Return:
<|think|>brief summary of what you learned.<\\think|>
<|final_answer|>the answer for the user.<\\final_answer|>

If an observation shows an action failed, reason about why it failed and try a corrected action.
For requests about current or latest information, verify the date and facts with actions.
Never present model memory, estimates, or unverified claims as current action results.
"""

    def history_cutoff_notice(self, last_unloaded_line: int) -> str:
        """Identify omitted physical lines in the current history file."""
        line_range = (
            f"lines 1-{last_unloaded_line} (1-based, inclusive)"
            if last_unloaded_line else "no earlier lines (empty range)"
        )
        return HISTORY_CUTOFF_NOTICE.format(
            history_file=json.dumps(str(self.history_file.resolve()), ensure_ascii=False),
            line_range=line_range,
        )

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
                        # Load only the latest active segment, including its notice.
                        # Keep old summary checkpoints readable, but never create them.
                        if item["role"] == "user" and item.get("state") in (
                            HISTORY_CUTOFF_STATE, "summary",
                        ):
                            history.clear()
                        if item["role"] == "user" and item.get("state") == HISTORY_CUTOFF_STATE:
                            # A resumed run may use a different working copy/path.
                            item["content"] = self.history_cutoff_notice(line_number - 1)
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
            with self.history_file.open("ab+") as file:
                # Imported JSONL may lack a trailing newline; keep records separate.
                file.seek(0, os.SEEK_END)
                if file.tell():
                    file.seek(-1, os.SEEK_END)
                    if file.read(1) not in {b"\n", b"\r"}:
                        file.write(b"\n")
                file.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            print(f"==== SYSTEM ==== Failed to save history: {exc}", flush=True)
            # A later cutoff must never discard messages that were not persisted.
            # Stop on any write/flush/fsync failure, not just a failed boundary.
            raise

    def history_context_size(self) -> float:
        """History size in K characters (1K = 1000), excluding system messages."""
        return sum(
            len(item["content"]) for item in self.memory if item["role"] != "system"
        ) / 1000

    def compact_history(self) -> bool:
        """Start a new context segment without summarizing or deleting history."""
        target = self.target_history_context_size
        before = self.history_context_size()
        if not target or before <= target:
            return False

        # Count physical lines, including blank/invalid records, not just messages.
        with self.history_file.open("r", encoding="utf-8") as file:
            last_unloaded_line = sum(1 for _ in file)
        notice = self.history_cutoff_notice(last_unloaded_line)
        # Persist the boundary before dropping anything from the model's context.
        self.save_history("user", notice, HISTORY_CUTOFF_STATE)
        self.memory = [m for m in self.memory if m["role"] == "system"]
        self.memory.append({"role": "user", "content": notice})
        print(
            f"==== SYSTEM ==== History cutoff: {before:g}K -> "
            f"{self.history_context_size():g}K; earlier history retained on disk",
            flush=True,
        )
        return True

    def run(self, text: str) -> None:
        now = datetime.now().astimezone()
        current_time = now.isoformat(timespec="seconds")
        self.run_id = now.strftime("%Y%m%dT%H%M%S%f%z")
        system_prompt = f"""You are an autonomous ReAct agent.

Current local date and time: {current_time}
Full history file (JSONL): {json.dumps(str(self.history_file.resolve()), ensure_ascii=False)}

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
        print(
            f"history_context_usage: {self.history_context_size():g}K; "
            f"target: {self.target_history_context_size}K",
            flush=True,
        )
        self.save_history("user", user_prompt, "user")
        state = "model"
        steps = 0
        response = ""

        while state != "done":
            if state == "model":
                if steps >= self.max_steps:
                    state = "max_steps"
                    continue

                self.compact_history()
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
                # Keep the terminal instruction after any new history boundary.
                self.compact_history()
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
    parser.add_argument("--effort", help="Reasoning level; defaults to REASONING_EFFORT or auto")
    parser.add_argument(
        "--target_history_context_size",
        "--target-history-context-size",
        type=int,
        default=DEFAULT_TARGET_HISTORY_CONTEXT_SIZE,
        help="Active history limit in K characters; exceeding it inserts a cutoff "
             "notice without summarizing (default: %(default)sK; 0 disables)",
    )
    args = parser.parse_args()
    agent = Agent(
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        history=args.history,
        effort=args.effort,
        target_history_context_size=args.target_history_context_size,
    )
    with open(args.input.strip(), "r") as f:
        text = f.read()
    agent.run(text)
