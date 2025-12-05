# llm_backend.py
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from llama_cpp import Llama


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatSession:
    system_prompt: str = "You are a helpful AI assistant."
    messages: List[Message] = field(default_factory=list)

    def reset(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str):
        self.messages.append(Message(role="assistant", content=content))

    def build_prompt(self) -> str:
        """
        Simple prompt format. You can change this to match your model’s style (e.g. LLaMA chat format).
        """
        parts = [f"System: {self.system_prompt}\n"]
        for msg in self.messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{prefix}: {msg.content}\n")
        parts.append("Assistant: ")
        return "".join(parts)


class LLMClient:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 6,
        n_gpu_layers: int = 0,  # set >0 if you have GPU support in llama.cpp build
    ):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

    def generate(
        self,
        chat: ChatSession,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[list] = None,
    ) -> str:
        if stop is None:
            stop = ["User:", "Assistant:", "System:"]

        prompt = chat.build_prompt()
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        text = result["choices"][0]["text"].strip()
        return text
