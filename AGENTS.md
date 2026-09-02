# AGENTS.md — mlx-timesfm

TimesFM 3.0 (Google Research) inference on Apple MLX.

## Read first

1. `.agents/SPEC.md` — what we're building, requirements, acceptance gates (A1–A4)
2. `.agents/PLAN.md` — phased plan + task tracker (update checkboxes as you go)
3. `tests/fixtures/real/golden_manifest.json` — frozen parity oracle provenance

## Hard rules

- **Never add `torch` or `safetensors` to project deps** (`pyproject.toml` /
  `uv.lock`). Package code and tests are MLX-only. MLX reads safetensors
  natively via `mx.load`.
- Do not add live cross-framework runners back to this repository. Established
  parity is frozen in committed golden fixtures and historical reports.
- Do not modify anything under `.references/` or `models/`.
- Project artifacts (parity reports, benchmarks, notes) go under `.agents/`.
- Parity is judged against SPEC.md tolerances — not "looks close".

## Commands

```bash
uv sync                      # deps (.venv)
uv run pytest                # tests (must pass torch-free)
uv run ruff check .          # lint
uv run python -c "import mlx.core as mx; ..."   # scratch experiments
```

## Layout

- `src/mlx_timesfm/` — package (src-layout, `uv_build` backend)
- `tests/` — MLX property, checkpoint, and golden regression tests
- `tests/fixtures/real/` — attributed real inputs + frozen golden forecasts
- `models/` — symlink → local weights (`models/timesfm_3_0/original/`)
- `.agents/` — SPEC.md, PLAN.md, parity-reports/, benchmarks/

## Environment notes

- mlx 0.32.2 + Metal verified working; default device is GPU
- weights: 445 tensors, 330.7M params, fp32, state-dict keys map 1:1 onto MLX modules
- parity is regression-tested on MLX CPU and GPU against frozen fp32 goldens
- watch-outs (details in SPEC §7): attention `scale=√head_dim`, additive mask
  direction, per-batch RoPE positions, MLX lazy-eval sync points, and
  RMSNorm eps: use the explicitly pinned fp32 reference value (SPEC R5)
