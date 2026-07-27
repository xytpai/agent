import os
import argparse
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
context_file = os.path.join(temp_dir, "context.txt")
_env = os.environ.copy()
_env["TORCH_CPP_LOG_LEVEL"] = "ERROR"


class Agent:
    def __init__(self, max_tokens: int, max_steps: int = 20):
        self.backend = get_backend()
        self.actions = ActionRunner()
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.memory = []
        self.react_note = """IMPORTANT:
You are running in a ReAct loop: Thought -> Action -> Observation -> Thought.
When you need to use a tool, output exactly one action and then stop immediately.
Never output a second Thought or Action before the system returns an Observation.
Never write Observation yourself; Observation is written only by the system after an action runs.
Never repeat or quote a previous Action block in your response.
After the closing Action Input fence, output nothing else and wait for Observation.
Use this format:
Thought: explain the next step briefly.
Action: one available action name
Action Input:
```text
arguments for the action
```

After the system returns an Observation, use it to decide the next step.
When the task is complete, do not call another action. Return:
Thought: brief summary of what you learned.
Final Answer: the answer for the user.

If an Observation shows an action failed, reason about why it failed and try a corrected action.
For requests about current or latest information, verify the date and facts with actions.
Never present model memory, estimates, or unverified claims as current action results.
"""

    def run(self, text: str, context: str = "") -> None:
        current_time = datetime.now().astimezone().isoformat(timespec="seconds")
        system_prompt = f"""You are an autonomous ReAct agent.

Current local date and time: {current_time}

Available actions:
{self.actions.desc()}

{self.react_note}
"""
        user_prompt = f"""Context:
{context or "(none)"}

User task:
{text}
"""
        self.memory = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        state = "model"
        steps = 0
        response = ""
        discarded_actions = 0

        while state != "done":
            if state == "model":
                if steps >= self.max_steps:
                    state = "max_steps"
                    continue

                steps += 1
                resp_chunks = []
                for chunk in self.backend.stream_response(self.memory, self.max_tokens):
                    resp_chunks.append(chunk)
                raw_response = "".join(resp_chunks)
                action_count = len(
                    list(self.actions.action_pattern.finditer(raw_response))
                )
                discarded_actions = max(0, action_count - 1)
                response = (
                    self.actions.trim_to_first_action(raw_response)
                    if action_count > 0
                    else raw_response
                )
                print(response, end="", flush=True)
                self.memory.append({"role": "assistant", "content": response})

                action_call = self.actions.parse_action(response)
                if "final answer:" in response.lower() and not action_call:
                    state = "done"
                else:
                    state = "action"

            elif state == "action":
                result = self.actions(response)
                if result is None:
                    state = "repair"
                else:
                    if discarded_actions:
                        result = (
                            f"[warning: discarded {discarded_actions} trailing "
                            f"Action block(s)]\n{result}"
                        )
                    discarded_actions = 0
                    observation = f"Observation: {result}"
                    print(f"\n\n{observation}\n", flush=True)
                    self.memory.append({"role": "user", "content": observation})
                    state = "model"

            elif state == "repair":
                observation = (
                    "No valid action was found. Continue with either a valid "
                    "ReAct action or a Final Answer."
                )
                print(f"\n\nObservation: {observation}\n", flush=True)
                self.memory.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
                state = "model"

            elif state == "max_steps":
                observation = (
                    f"Reached max_steps={self.max_steps}. Stop and provide the best final "
                    "answer from the collected observations."
                )
                print(f"\n\nObservation: {observation}\n", flush=True)
                self.memory.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )

                resp_chunks = []
                for chunk in self.backend.stream_response(self.memory, self.max_tokens):
                    resp_chunks.append(chunk)
                raw_response = "".join(resp_chunks)
                action_count = len(
                    list(self.actions.action_pattern.finditer(raw_response))
                )
                response = (
                    self.actions.trim_to_first_action(raw_response)
                    if action_count > 0
                    else raw_response
                )
                print(response, end="", flush=True)
                self.memory.append({"role": "assistant", "content": response})
                state = "done"

        print("", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AToy: An agent create shit")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--input", type=str, default="None")
    args = parser.parse_args()
    agent = Agent(max_tokens=args.max_tokens, max_steps=args.max_steps)
    with open(args.input.strip(), "r") as f:
        text = f.read()
    agent.run(text)
