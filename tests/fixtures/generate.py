#!/usr/bin/env python
"""A2 fixture generator (SPEC §5 A2).

Five deterministic series, written as committed CSVs next to this file. CSV —
never npz (npz is gitignored). Re-running with the same seeds reproduces the
committed files byte-for-byte; CI check: `git diff --exit-code tests/fixtures/`.

Fixtures:
  1. trend.csv       linear trend + N(0, 0.1) noise          (len 1024)
  2. sine.csv        sine, period 24, amplitude 1            (len 1024)
  3. white_noise.csv standard normal                          (len 1024)
  4. near_flat.csv   |x| <= 1e-3 with one step at t=100       (len 1024)
  5. ar1_mv3.csv     3 correlated AR(1) channels              (len 1536)

Lengths cover A2 runs: context 512 + horizon up to 512 for the univariate
fixtures; context 1024 + horizon up to 512 for the multivariate set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEN_UNI = 1024  # ctx 512 + max horizon 512
LEN_MV = 1536  # ctx 1024 + max horizon 512


def trend(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(LEN_UNI, dtype=np.float64)
    return 2.0 + 0.01 * t + rng.normal(0.0, 0.1, LEN_UNI)


def sine(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(LEN_UNI, dtype=np.float64)
    return np.sin(2 * np.pi * t / 24.0)


def white_noise(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, LEN_UNI)


def near_flat(rng: np.random.Generator) -> np.ndarray:
    # Hard A2 case: sigma ~ 1e-3 makes the 2e-3*sigma bound tightest.
    x = np.clip(rng.normal(0.0, 1e-4, LEN_UNI), -1e-3, 1e-3)
    x[100:] += 2.0  # one step at t=100
    return x


def ar1_mv3(rng: np.random.Generator) -> np.ndarray:
    # 3 correlated AR(1): each channel = shared driver (weight rho) + own
    # independent AR(1), so corr(ch_i, ch_j) ≈ rho² for i≠j.
    phi, sigma, rho = 0.7, 0.1, 0.6
    shared = _ar1(rng, LEN_MV, phi, sigma)
    out = np.stack(
        [rho * shared + np.sqrt(1.0 - rho**2) * _ar1(rng, LEN_MV, phi, sigma) for _ in range(3)],
        axis=1,
    )
    return out


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


FIXTURES = {
    "trend.csv": (trend, 1, ["value"]),
    "sine.csv": (sine, 2, ["value"]),
    "white_noise.csv": (white_noise, 3, ["value"]),
    "near_flat.csv": (near_flat, 4, ["value"]),
    "ar1_mv3.csv": (ar1_mv3, 5, ["v0", "v1", "v2"]),
}


def main() -> None:
    for name, (fn, seed, cols) in FIXTURES.items():
        data = fn(np.random.default_rng(seed))
        np.savetxt(
            HERE / name,
            data,
            delimiter=",",
            header=",".join(cols),
            comments="",
            fmt="%.9f",
        )
        print(f"wrote {name} {data.shape}")


if __name__ == "__main__":
    main()
