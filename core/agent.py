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
        self.memory = []

    def user_ask(self, text: str, context: str = "") -> None:
        prompt = f"""Information:
{self.actions.desc()}

Context:
{context}

Answer:
{text}
"""
        self.memory = [prompt]

    def step(self) -> None:
        prompt = "\n\n".join(self.memory)
        resp = self.backend.get_response(prompt, self.max_tokens, stream_print=True)
        self.memory.append(resp)

    def maybe_take_action(self):
        res = self.actions(self.memory[-1])
        if res:
            res_ = f"\nSYSTEM-ACTION_RESULTS: {res}\n"
            print(res_, flush=True)
            self.memory.append(res_)
            return res
        else:
            return None

    def run(self, text: str):
        self.user_ask(text)
        while True:
            self.step()
            if self.maybe_take_action():
                self.step()
            else:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AToy: An agent create shit")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--input", type=str, default="None")
    args = parser.parse_args()
    agent = Agent(max_tokens=args.max_tokens)
    with open(args.input.strip(), "r") as f:
        text = f.read()
    agent.run(text)
