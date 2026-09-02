# SPEC — mlx-timesfm

> Status: draft v1 (2026-09-01). This file defines WHAT we are building and how we
> judge it done. HOW / when lives in `PLAN.md`. Task tracking happens in PLAN.md.

## 1. Objective

Port Google Research **TimesFM 3.0** (inference-only PyTorch reference at
`.references/timesfm/src/timesfm3/`) to **Apple MLX**, so the model runs natively
on Apple silicon (Metal GPU) with results that are numerically on par with the
PyTorch reference.

## 2. Background & inputs (already verified)

- Reference implementation: `.references/timesfm/src/timesfm3/` (source is
  Apache-2.0; the **pretrained weights carry a separate Google terms license** —
  see the HF repo), PyTorch, inference-only (core ≈2k LoC: model / transformer /
  dense / normalization / util / cpm_revin_refine / configs; plus a ~760-LoC
  forecaster wrapper we may port last or skip).
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
- **F4 — Output parity**: quantile forecasts match the PyTorch reference within
  Section 5 tolerances, on the same inputs.
- **F5 — Python API** (package `mlx_timesfm`):
  ```python
  model = mlx_timesfm.load("models/timesfm_3_0/original")   # path or HF-style id later
  preds = model.forecast(target, horizon, ...)             # (b, v, horizon, q) quantiles
  ```
  Plus a thin sklearn-ish `TimesFM3Forecaster` wrapper (optional, P2).
- **F6 — dtype**: fp32 is the supported baseline; bf16/fp16 inference may be
  offered as opt-in once fp32 parity is met.

## 4. Non-functional requirements

- **N1 — Dependency red line**: package deps are MLX-only. `torch` and
  `safetensors` are **never** in `pyproject.toml` / `uv.lock` (now or later).
  Parity tooling may use torch only inside the separate, gitignored `.venv-torch/`.
- **N2 — Environment**: uv-managed, Python 3.13, `mlx>=0.32.2` (current latest),
  src-layout (`src/mlx_timesfm/`), `uv_build` backend.
- **N3 — Performance**: primary target is the absolute A3 throughput target;
  secondary context: wall time vs the torch-**CPU** reference (we deliberately do
  not baseline against torch-MPS: SDPA masked-attention support there is patchy
  and may silently CPU-fallback, making the comparison meaningless); peak memory
  ≤ weights + O(b·v·n·d) activations, fp32 weights fit in 1.3 GB.
- **N4 — Quality**: typed, tested (property tests + parity tests), ruff-clean.

## 5. Acceptance criteria (parity definition)

Parity harness runs both stacks on identical inputs; torch side lives in
`.venv-torch/` (torch + editable install of the reference clone), mlx side in `.venv/`;
a small bridge script exchanges tensors via `.npz` files.

- **A1 (gates each phase)** — layer/stack/forward alignment: **both stacks on
  CPU** (torch CPU; mlx via `mx.set_default_device(mx.cpu)`), fp32, on
  **identical weights and inputs** (npz bridge; random or real weights — the
  model is deterministic at inference). Same-device on purpose: a diff must
  implicate code, not different kernels. Max abs diff ≤ 1e-4 on activations
  (RoPE, MHA, MixingTransformer, full forward), ≤ 1e-5 on normalized/reduced
  quantities (RMSNorm outputs, running stats). GPU behavior is covered by A2.
- **A2 (end-to-end, fp32)** — real weights, torch-CPU vs mlx-**GPU** (this gate
  intentionally crosses devices; tolerance accounts for kernel differences).
  **Fixtures** (committed as CSV + a seeded generator script under
  `tests/fixtures/` — never npz, which is gitignored): five deterministic
  series — (1) linear trend + N(0,0.1) noise, (2) sine, period 24, amp 1,
  (3) white noise, (4) near-flat (|x|≤1e-3) with one step at t=100,
  (5) multivariate: 3 correlated AR(1) channels. Context 512 (1024 for the
  multivariate fixture set), horizons 32/128/512.
  **σ := population std of that fixture's unmasked context values** (per target
  series). Criteria: per-quantile max abs diff ≤ 2e-3·σ, and quantile ordering
  matches the reference (the port must not introduce new quantile crossings;
  crossings already present in torch output are allowed).
  If mlx runs on CPU here, an additional tighter report (≤5e-4·σ) is recorded
  as informational.
- **A3 (perf smoke)** — decode of 1000 univariate series (context 512, h=128)
  completes in < 60 s on GPU; report logged to `.agents/benchmarks/`.
- **A4** — `uv run pytest` green in a clean clone (torch-free); parity tests
  carry a `parity` marker and skip cleanly when `.venv-torch/` is absent.

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
- **R5 RMSNorm eps defaults are framework- and version-specific** — do not cite
  or rely on any default value. Rule: the port takes an explicit `eps` knob and
  parity runs instantiate torch RMSNorm with the **same eps the reference torch
  env actually uses** (probe `torch.nn.RMSNorm(80).eps` at harness setup, record
  it in every parity report). Otherwise A1's norm gate silently measures eps
  instead of code.
- **R6 weight-key round-trip**: module attribute tree must produce exactly the
  445 safetensors keys (names, shapes). Loading must fail loudly on any
  missing/extra key or shape mismatch rather than silently skipping tensors.
