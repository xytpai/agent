import argparse
import os
from abc import ABC, abstractmethod
from typing import Iterator
from openai import OpenAI

EFFORT_LEVELS = ("auto", "none", "minimal", "low", "medium", "high", "xhigh", "max")


def resolve_effort(effort: str | None = None) -> str | None:
    if effort is None:
        effort = os.environ.get("REASONING_EFFORT", "auto")
    effort = effort.strip().lower()
    if effort == "ultra-high":
        effort = "xhigh"
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"Invalid effort {effort!r}; choose from {', '.join(EFFORT_LEVELS)}")
    return None if effort == "auto" else effort


class AgentBackend(ABC):
    def __init__(self, effort: str | None = None):
        self.reasoning_effort = resolve_effort(effort)
        self.initialize()

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def stream_response(self, inputs: list, max_tokens: int) -> Iterator[str]:
        # inputs = [{"role": "user", "content": content}, ...]
        pass


class OpenaiBackend(AgentBackend):
    def initialize(self):
        BASE_URL = os.environ.get("BASE_URL")
        API_KEY = os.environ.get("API_KEY")
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key="dummy",
            default_headers={"Ocp-Apim-Subscription-Key": API_KEY},
        )
        self.model = os.environ.get("MODEL_NAME")

    def stream_response(self, inputs: list, max_tokens: int = 65536) -> Iterator[str]:
        options = {}
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort
        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=inputs,
            stream=True,
            **options,
        )
        try:
            for event in stream:
                if len(event.choices) > 0:
                    delta = event.choices[0].delta
                    if delta and delta.content and len(delta.content) > 0:
                        yield delta.content
        finally:
            close_stream = getattr(stream, "close", None)
            if close_stream:
                close_stream()


def get_backend(effort: str | None = None):
    model_name = os.environ.get("MODEL_NAME")
    if not model_name:
        raise ValueError("MODEL_NAME must be set")
    if "gpt" in model_name.lower():
        backend = OpenaiBackend(effort=effort)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    print(
        f"==== SYSTEM ==== MODEL_NAME:{model_name} "
        f"REASONING_EFFORT:{backend.reasoning_effort or 'auto (provider default)'}\n",
        flush=True,
    )
    return backend


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model backend")
    parser.add_argument("text")
    parser.add_argument("--effort", help="Reasoning level; defaults to REASONING_EFFORT or auto")
    args = parser.parse_args()
    backend = get_backend(effort=args.effort)
    inputs = [
        {"role": "user", "content": args.text},
    ]
    for chunk in backend.stream_response(inputs, 65536):
        print(chunk, end="", flush=True)
    print("", flush=True)
