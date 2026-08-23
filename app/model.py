"""The GPU seam.

Everything above this module (extract, validate, parsing, boxes, the API,
the eval harness) talks to a VLMEngine and never imports torch. That keeps
the entire pipeline developable and testable on a machine with no GPU;
only QwenEngine needs the T4.

It also makes batching the native interface rather than a special case:
generate() takes a list and returns a list, which is what lets the
self-consistency and grounding passes ride along at almost no wall-clock
cost (spec D5).
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class GenerationRequest:
    image: Image.Image
    prompt: str


@runtime_checkable
class VLMEngine(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def device(self) -> str: ...

    def generate(self, requests: list[GenerationRequest]) -> list[str]: ...


@dataclass
class FakeEngine:
    """Test double. Either a flat list of replies consumed in order across
    calls, or a callable that maps a batch to replies."""

    responses: list[str] | Callable[[list[GenerationRequest]], list[str]]
    calls: list[list[GenerationRequest]] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def device(self) -> str:
        return "cpu"

    def generate(self, requests: list[GenerationRequest]) -> list[str]:
        self.calls.append(list(requests))
        if callable(self.responses):
            return self.responses(requests)
        assert len(self.responses) >= len(requests), (
            f"FakeEngine script exhausted: {len(requests)} requested, "
            f"{len(self.responses)} left"
        )
        out, self.responses = self.responses[: len(requests)], self.responses[len(requests):]
        return out
