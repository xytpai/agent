import os
import sys
import httpx
from abc import ABC, abstractmethod
from anthropic import Anthropic
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
    def get_response(self, content: str, max_tokens: int, stream_print: bool):
        pass


class AnthropicBackend(AgentBackend):
    def initialize(self):
        self.client = Anthropic(
            base_url=BASE_URL,
            api_key="dummy",
            default_headers={"Ocp-Apim-Subscription-Key": API_KEY}
        )
        self.model = MODEL_NAME
    
    def get_response(self, content: str, max_tokens: int = 65536, stream_print: bool = True) -> str:
        if not stream_print:
            if max_tokens <= 16384:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}]
                )
                out_str = response.content[0].text.strip()
            else:
                out_str = ""
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}]
                ) as stream:
                    for text in stream.text_stream:
                        out_str += text
        else:
            out_str = ""
            with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}]
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        out_str += text
            return out_str
        return out_str


class OpenaiBackend(AgentBackend):
    def initialize(self):
        BASE_URL = os.environ.get("BASE_URL")
        API_KEY = os.environ.get("API_KEY")
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key="dummy",
            default_headers={"Ocp-Apim-Subscription-Key": API_KEY}
        )
        self.model = MODEL_NAME
    
    def get_response(self, content: str, max_tokens: int = 65536, stream_print: bool = True) -> str:
        if not stream_print:
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": content}]
            )
            out_str = response.choices[0].message.content.strip()
        else:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
                stream=True,
            )
            out_chunks = []
            for event in stream:
                if len(event.choices) > 0:
                    delta = event.choices[0].delta
                    if delta and delta.content and len(delta.content) > 0:
                        piece = delta.content
                        out_chunks.append(piece)
                        print(piece, end="", flush=True)
            out_str = "".join(out_chunks).strip()
        return out_str


def get_backend():
    global MODEL_NAME
    print(f"==== SYSTEM ==== MODEL_NAME:{MODEL_NAME}\n", flush=True)
    if 'gpt' in MODEL_NAME.lower():
        return OpenaiBackend()
    else:
        return AnthropicBackend()


if __name__ == '__main__':
    backend = get_backend()
    backend.get_response(sys.argv[1], stream_print=True)
