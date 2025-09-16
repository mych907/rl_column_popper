from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a numpy Generator with PCG64 bit generator.

    If seed is None, entropy is drawn from OS.
    """
    if seed is None:
        return np.random.Generator(np.random.PCG64())
    return np.random.Generator(np.random.PCG64(seed))


def _hash_to_uint32(name: str) -> int:
    h = hashlib.sha256(name.encode("utf-8")).digest()
    # Take first 4 bytes as little-endian uint32
    return int.from_bytes(h[:4], "little", signed=False)


@dataclass
class RngPool:
    """Deterministic pool of child RNGs derived from a root seed.

    Each key maps to its own independent child generator using a stable hash.
    """

    seed: int | None

    def __post_init__(self) -> None:
        self._cache: dict[str, np.random.Generator] = {}
        # Maintain a root SeedSequence for documentation; child derivation uses hashing
        self._root_ss = np.random.SeedSequence(self.seed) if self.seed is not None else None

    def get(self, key: str) -> np.random.Generator:
        if key in self._cache:
            return self._cache[key]
        # Derive deterministic child seed from root seed and key hash
        if self.seed is None:
            child_seed = None
        else:
            child_seed = (int(self.seed) ^ _hash_to_uint32(key)) & 0xFFFFFFFF
        gen = make_rng(child_seed)
        self._cache[key] = gen
        return gen


def split_generators(seed: int, n: int) -> list[np.random.Generator]:
    """Split a seed into N independent child generators using SeedSequence.spawn."""
    ss = np.random.SeedSequence(seed)
    return [np.random.Generator(np.random.PCG64(s)) for s in ss.spawn(n)]


__all__ = ["make_rng", "RngPool", "split_generators"]
