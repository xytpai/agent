import os
import sys
import httpx
from abc import ABC, abstractmethod
from typing import Iterator
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")


class AgentBackend(ABC):
    def __init__(self):
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
        self.model = MODEL_NAME

    def stream_response(self, inputs: list, max_tokens: int = 65536) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=inputs,
            stream=True,
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


def get_backend():
    global MODEL_NAME
    print(f"==== SYSTEM ==== MODEL_NAME:{MODEL_NAME}\n", flush=True)
    if "gpt" in MODEL_NAME.lower():
        return OpenaiBackend()
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")


if __name__ == "__main__":
    backend = get_backend()
    text = sys.argv[1]
    inputs = [
        {"role": "user", "content": text},
    ]
    for chunk in backend.stream_response(inputs, 65536):
        print(chunk, end="", flush=True)
    print("", flush=True)
