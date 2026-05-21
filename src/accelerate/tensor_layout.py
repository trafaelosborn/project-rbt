"""
Contiguous fingerprint tensor packing for the Fortran acceleration path.

The tensor is represented as one 1D float64 buffer plus a deterministic layout.
Each component can be viewed without additional copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorComponent:
    name: str
    shape: tuple[int, ...]
    offset: int
    order: str = "F"

    @property
    def size(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def stop(self) -> int:
        return self.offset + self.size

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "offset": self.offset,
            "stop": self.stop,
            "order": self.order,
        }


class FingerprintTensorLayout:
    """
    Single-buffer layout for the full fingerprint state.

    The buffer layout is:
        1. cooccurrence matrix        [vocab, vocab]
        2. positional matrix          [vocab, positional_width]
        3. bigram profile vector      [bigram_profile_size]
        4. trigram profile vector     [trigram_profile_size]
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        positional_width: int,
        bigram_profile_size: int,
        trigram_profile_size: int,
        dtype: np.dtype = np.float64,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if positional_width <= 0:
            raise ValueError("positional_width must be positive")
        if bigram_profile_size < 0 or trigram_profile_size < 0:
            raise ValueError("n-gram profile sizes cannot be negative")

        self.vocab_size = int(vocab_size)
        self.positional_width = int(positional_width)
        self.bigram_profile_size = int(bigram_profile_size)
        self.trigram_profile_size = int(trigram_profile_size)
        self.dtype = np.dtype(dtype)

        offset = 0
        components = []
        for name, shape in (
            ("cooccurrence", (self.vocab_size, self.vocab_size)),
            ("positional", (self.vocab_size, self.positional_width)),
            ("bigram_profile", (self.bigram_profile_size,)),
            ("trigram_profile", (self.trigram_profile_size,)),
        ):
            component = TensorComponent(name=name, shape=shape, offset=offset)
            components.append(component)
            offset = component.stop

        self._components = tuple(components)
        self._by_name = {component.name: component for component in self._components}
        self.total_size = offset

    @property
    def components(self) -> tuple[TensorComponent, ...]:
        return self._components

    def allocate(self) -> np.ndarray:
        return np.empty(self.total_size, dtype=self.dtype)

    def view(self, buffer: np.ndarray, name: str) -> np.ndarray:
        if name not in self._by_name:
            raise KeyError(f"Unknown tensor component: {name}")
        arr = np.asarray(buffer)
        if arr.ndim != 1:
            raise ValueError(f"Tensor buffer must be 1D, got {arr.ndim}D")
        if arr.dtype != self.dtype:
            raise ValueError(f"Tensor buffer dtype must be {self.dtype}, got {arr.dtype}")
        if arr.size != self.total_size:
            raise ValueError(f"Tensor buffer size must be {self.total_size}, got {arr.size}")

        component = self._by_name[name]
        return arr[component.offset : component.stop].reshape(component.shape, order=component.order)

    def pack(
        self,
        *,
        cooccurrence: np.ndarray,
        positional: np.ndarray,
        bigram_profile: np.ndarray,
        trigram_profile: np.ndarray,
    ) -> np.ndarray:
        buffer = self.allocate()
        inputs = {
            "cooccurrence": np.asarray(cooccurrence, dtype=self.dtype),
            "positional": np.asarray(positional, dtype=self.dtype),
            "bigram_profile": np.asarray(bigram_profile, dtype=self.dtype),
            "trigram_profile": np.asarray(trigram_profile, dtype=self.dtype),
        }

        for name, value in inputs.items():
            view = self.view(buffer, name)
            if value.shape != view.shape:
                raise ValueError(f"{name} shape mismatch: expected {view.shape}, got {value.shape}")
            np.copyto(view, value)
        return buffer

    def manifest(self) -> dict:
        return {
            "dtype": str(self.dtype),
            "total_size": self.total_size,
            "components": [component.to_dict() for component in self._components],
        }


__all__ = ["FingerprintTensorLayout", "TensorComponent"]

