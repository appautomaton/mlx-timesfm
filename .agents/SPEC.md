# SPEC — mlx-timesfm

> Status: draft v1 (2026-09-01). This file defines WHAT we are building and how we
> judge it done. HOW / when lives in `PLAN.md`. Task tracking happens in PLAN.md.

## 1. Objective

Run Google Research **TimesFM 3.0** inference natively in **Apple MLX** on Apple
silicon. Numerical behavior is locked to the already-validated original
implementation through committed, versioned golden forecasts.

## 2. Background & inputs (already verified)

- Historical reference provenance is pinned in the golden manifest and parity
  reports. It is not installed or executed by this project. The **pretrained
  weights carry a separate Google terms license**; see the checkpoint license.
- Weights: `models/timesfm_3_0/original/model.safetensors` — 445 tensors,
  330,710,976 params, float32, PyTorch state-dict naming. Verified loadable via
  `mx.load()` (MLX reads safetensors natively). Linear weights are `(out, in)`,
  same as `mlx.nn.Linear` → **zero conversion** expected; `model.update()` directly.
- Model config: `models/timesfm_3_0/original/config.json`
  (20 layers, d=1280, heads=16 → head_dim=80, RMSNorm, qk_norm=rms, v_norm=none,
  RoPE on seq axis only, causal seq-attn, non-causal variate-attn, ReLU FFN,
  input_patch=32, output_patch=64, 9 quantiles, stitching, linear detrending
  threshold 0.5, iterative CPM-RevIN refine).

## 3. Functional requirements

- **F1 — Model core**: `forward()` equivalent of `TimesFM3Torch.forward`
  (patched inputs `(b, v, n, p)` → logits `(b, v, n, o, q)`), including RevIN
  running stats, roll-based future-covariate patches, ResBlock, transformer stack,
  output head, CPM-RevIN refinement, inverse RevIN + clipping.
- **F2 — Decode API**: `decode()` equivalent — context/horizon padding, linear
  detrending with covariate groups, horizon CPM masking, stitching (or chunk)
  extraction, trend re-addition. Supports univariate, multivariate, past-only and
  past-future covariates.
- **F3 — Weights loading**: load `config.json` + safetensors directly from the
  `original/` directory into an `nn.Module`; no intermediate converted artifact
  is required (a cached converted copy is allowed as an optimization).
- **F4 — Output parity**: quantile forecasts match the frozen reference goldens
  within Section 5 tolerances on the same inputs.
- **F5 — Python API** (package `mlx_timesfm`):
  ```python
  model = mlx_timesfm.load("models/timesfm_3_0/original")   # path or HF-style id later
  preds = model.forecast(target, horizon, ...)             # (b, v, horizon, q) quantiles
  ```
  Plus a thin sklearn-ish `TimesFM3Forecaster` wrapper (optional, P2).
- **F6 — dtype**: fp32 is the supported baseline, including full-fp32 matrix
  kernels (`MLX_ENABLE_TF32=0` unless explicitly overridden); bf16/fp16 or
  reduced-precision fp32 inference may be offered as opt-in once fp32 parity
  is met.

## 4. Non-functional requirements

- **N1 — Dependency red line**: package code, tests, and project environments
  are MLX-only. `torch` and `safetensors` are **never** project dependencies or
  imports. Live cross-framework parity runners are retired.
- **N2 — Environment**: uv-managed, Python 3.13, `mlx>=0.32.2` (current latest),
  src-layout (`src/mlx_timesfm/`), `uv_build` backend.
- **N3 — Performance**: primary target is the absolute A3 throughput target.
  Peak memory ≤ weights + O(b·v·n·d) activations; fp32 weights fit in 1.3 GB.
- **N4 — Quality**: typed, tested (property tests + parity tests), ruff-clean.

## 5. Acceptance criteria (parity definition)

Initial parity was established live and recorded under
`.agents/parity-reports/`. Ongoing regression is MLX-only against committed
CSV goldens whose source revision, checkpoint/config hashes, execution profile,
shape, and tolerance are pinned in `tests/fixtures/real/golden_manifest.json`.

- **A1 (operator/property regression)** — fp32 operator, masking, normalization,
  attention, stack, preprocessing, and decode properties pass on MLX.
- **A2 (end-to-end golden parity, fp32)** — original weights on MLX CPU and GPU,
  using the committed real multivariate fixture (context 512, horizon 128,
  targets plus past-only and past/future covariates). Per-target max absolute
  difference must be ≤2e-3·σ, where σ is the population standard deviation of
  the context. The port must introduce no adjacent-quantile crossing whose
  golden gap exceeds 4e-3·σ. Full-fp32 matrix kernels are required.
- **A3 (perf smoke)** — decode of 1000 univariate series (context 512, h=128)
  completes in < 60 s on GPU; report logged to `.agents/benchmarks/`.
- **A4** — `uv run pytest` green in a clean MLX-only clone. Tests requiring the
  separately distributed original checkpoint skip cleanly when it is absent.

## 6. Out of scope

Training / fine-tuning; JAX-Flax path; timesfm ≤ 2.5 and `v1/`; GPU export /
mlx export format; distributed; HF hub upload. (May enter future specs.)

## 7. Risks

- **R1 attention scale semantics** (`scale=√head_dim` + PerDimScale): the #1
  silent-mismatch candidate; gated by A1 at MHA level.
- **R2 mask semantics inversion** (True=attend vs additive −1e9; leading-only
  cumprod masking) — gated by A1 at stack level.
- **R3 RoPE with arbitrary per-batch positions** — MLX `nn.RoPE.apply` only takes
  an int offset; custom implementation required; gated by A1.
- **R4 lazy-eval semantics** (`torch.where`-style branching, `.item()` sync points
  in KV-cache) — may surface as perf, not correctness, issues.
- **R5 RMSNorm eps** — never rely on a framework default. The established fp32
  reference value (`1.1920928955078125e-7`) is explicit in package config and
  the golden manifest.
- **R6 weight-key round-trip**: module attribute tree must produce exactly the
  445 safetensors keys (names, shapes). Loading must fail loudly on any
  missing/extra key or shape mismatch rather than silently skipping tensors.
