import os
import argparse
from pathlib import Path
from backends import get_backend
from actions import ActionRunner

base_dir = Path(__file__).resolve().parent
temp_dir = base_dir / "../temp"
temp_dir.mkdir(parents=True, exist_ok=True)
context_file = os.path.join(temp_dir, "context.txt")
_env = os.environ.copy()
_env["TORCH_CPP_LOG_LEVEL"] = "ERROR"


class Agent:
    def __init__(self, max_tokens: int):
        self.backend = get_backend()
        self.actions = ActionRunner()
        self.max_tokens = max_tokens
        self.end_pattern = "[[[END]]]<<<<<>>>>>"
        self.memory = []

    def initialize(self, text: str, context: str = "") -> None:
        prompt = f"""Information:
{self.actions.desc()}

Context:
{context}

Answer:
{text}

Think before taking action. Take a summary when you got the final answer.
Return {self.end_pattern} when you think the entire conversation can be ended.
NOTE: when you need to take an action, first return the action string directly.
If no useful information in SYSTEM-ACTION_RESULTS, think about why your action failed and try again !!!
"""
        self.memory = [prompt]

    def step(self) -> None:
        prompt = "\n\n".join(self.memory)
        resp = self.backend.get_response(prompt, self.max_tokens, stream_print=True)
        self.memory.append(resp)

    def maybe_take_action(self):
        res = self.actions(self.memory[-1])
        if res:
            res_ = f"\n\nSYSTEM-ACTION_RESULTS: {res}\n\n"
            print(res_, flush=True)
            self.memory.append(res_)
            return res
        else:
            return None

    def is_end(self) -> bool:
        return self.end_pattern in self.memory[-1]

    def run(self, text: str):
        self.initialize(text)
        self.step()
        while True:
            if self.is_end():
                print("", flush=True)
                return
            self.maybe_take_action()
            self.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AToy: An agent create shit")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--input", type=str, default="None")
    args = parser.parse_args()
    agent = Agent(max_tokens=args.max_tokens)
    with open(args.input.strip(), "r") as f:
        text = f.read()
    agent.run(text)
