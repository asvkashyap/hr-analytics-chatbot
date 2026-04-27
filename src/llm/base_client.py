"""
Abstract base class for LLM clients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    model: str = ""
    raw: Any = field(default=None, repr=False)

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_summary(self) -> str:
        return (
            f"tokens={self.total_tokens:,} "
            f"(in={self.input_tokens:,}, out={self.output_tokens:,}) | "
            f"cost=${self.total_cost_usd:.5f}"
        )


class BaseLLMClient(ABC):

    @abstractmethod
    def complete(self, prompt: str, temperature: float | None = None) -> LLMResponse:
        ...

    @abstractmethod
    def health_check(self) -> bool:
        ...