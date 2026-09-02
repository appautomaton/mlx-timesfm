# AGENTS.md — mlx-timesfm

TimesFM 3.0 (Google Research) ported from PyTorch to Apple MLX.

## Read first

1. `.agents/SPEC.md` — what we're building, requirements, acceptance gates (A1–A4)
2. `.agents/PLAN.md` — phased plan + task tracker (update checkboxes as you go)
3. `.references/timesfm/src/timesfm3/` — read-only PyTorch source of truth

## Hard rules

- **Never add `torch` or `safetensors` to project deps** (`pyproject.toml` /
  `uv.lock`). The package is MLX-only. MLX reads safetensors natively via `mx.load`.
- Torch-based parity/reference code goes in the separate, gitignored
  `.venv-torch/` env only; package code must never import torch.
- Do not modify anything under `.references/` or `models/`.
- Project artifacts (parity reports, benchmarks, notes) go under `.agents/`.
- Parity is judged against SPEC.md tolerances — not "looks close".

## Commands

```bash
uv sync                      # deps (.venv)
uv run pytest                # tests (must pass torch-free)
uv run ruff check .          # lint
uv run python -c "import mlx.core as mx; ..."   # scratch experiments
# parity reference env (one-time; uv venvs contain no pip — use uv pip):
#   uv venv .venv-torch --python 3.13
#   uv pip install --python .venv-torch/bin/python -e ".references/timesfm[torch]"
```

## Layout

- `src/mlx_timesfm/` — package (src-layout, `uv_build` backend)
- `tests/` — property tests; `tests/parity/` — torch↔mlx npz bridge
- `models/` — symlink → local weights (`models/timesfm_3_0/original/`)
- `.agents/` — SPEC.md, PLAN.md, parity-reports/, benchmarks/

## Environment notes

- mlx 0.32.2 + Metal verified working; default device is GPU
- weights: 445 tensors, 330.7M params, fp32, state-dict keys map 1:1 onto MLX modules
- parity gate A1 runs **both stacks on CPU** (kernel-neutral); GPU is tested at A2
- watch-outs (details in SPEC §7): attention `scale=√head_dim`, additive mask
  direction, per-batch RoPE positions, MLX lazy-eval sync points, and
  RMSNorm eps: never trust either framework's default — pass eps explicitly and
  probe the torch env's actual value (SPEC R5)
