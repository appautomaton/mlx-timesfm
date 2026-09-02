# PLAN — mlx-timesfm

> Companion to `SPEC.md` (WHAT/done-criteria). This file is the HOW and the
> task tracker. Gate refs (A1/A2…) and gotcha refs are defined in SPEC.md.
> Reference code: `.references/timesfm/src/timesfm3/` (read-only source of truth).

## Weight layout cheat-sheet (verified)

- Keys = PyTorch state_dict names, no outer prefix; `mx.load` → `model.update()` directly.
- `pre_transformer_resblock.{hidden,residual}_layer.weight (1280,192)`, `output_layer (1280,1280)`
- per layer ×20: `{seq,var}_attn.{query,key,value,out}_proj (1280,1280)`,
  `{query,key}_ln.weight (80,)`, `per_dim_scale.per_dim_scale (80,)`,
  6× `*_ln.weight (1280,)`, `ff{0,1} (1280,1280)`
- `output_head.weight (576,1280)` + **only bias in the model** `(576,)`
- Module attribute names must mirror these key names exactly — enforce with a
  round-trip test: `sorted(flatten(model.parameters())) == sorted(safetensors keys)`
  and matching shapes/dtypes; `load()` must **raise** on any missing/extra key or
  shape mismatch (R6), never silently partial-load.
- **RMSNorm: always `eps=1e-5`** (torch default; mlx default is 1e-6 — R5).
- Do NOT port torch's lazy `ResidualBlock.set_input_dims` hack — take
  `input_dim=2*(ip+op)=192` in the constructor.

## Phase 0 — Scaffolding ✅ done

- [x] `uv init --lib --vcs git`, Python 3.13, `mlx==0.32.2` (+mlx-metal), src-layout
- [x] Dependency red line recorded (N1) — torch only ever in `.venv-torch/`
- [x] `.gitignore` (.venv*, .references/, models symlink), initial commit

## Phase 1a — Parity harness + config loader

Goal: infrastructure to prove every operator exists before any operator is ported.

- [ ] `.venv-torch/`: `uv venv .venv-torch --python 3.13` — NOTE: uv-created venvs
      have **no pip inside**; install via
      `uv pip install --python .venv-torch/bin/python -e ".references/timesfm[torch]"`
      (torch comes via the `[torch]` extra — explicit, not by luck; also pulls
      huggingface_hub/safetensors/numpy, fine inside this isolated env;
      `import timesfm3` requires torch and gets it here)
- [ ] `uv add --dev pytest ruff` (main .venv dev deps — torch is never one of them)
- [ ] `config.py`: parse `models/timesfm_3_0/original/config.json` → dataclasses
      mirroring `.references` `configs.py` semantics; single source of truth for
      model construction, tests, and `mlx_timesfm.load()`
- [ ] `tests/parity/bridge.py`: a seeded generator writes **inputs AND model
      weights** to npz so both stacks start bit-identical; torch runner + mlx
      runner each emit outputs; comparator diffs into
      `.agents/parity-reports/*.md`. npz artifacts → `tests/parity/artifacts/`
      (gitignored). Both runners **force CPU** (A1).
- [ ] A2 fixture generator: 5 deterministic series (SPEC A2 list) →
      `tests/fixtures/*.csv` + `tests/fixtures/generate.py` (committed; not npz)
- [ ] pytest marker `parity` — auto-skip when `.venv-torch/` is absent (A4)
- [ ] Smoke: bridge round-trips one trivial op (e.g. `mx.maximum` vs `F.relu`)

## Phase 1b — Pointwise operators

- [ ] `normalization.py`: `PerDimScale` (init ⇒ scale ≈ 1/√d)
- [ ] `transformer.py`: `rope()` — half-rotation style, 3D/4D inputs, arbitrary
      `(b,n)` int positions (R3); `make_attn_mask` / `make_segment_mask` →
      additive −1e9 float masks (R2)
- [ ] Gate **A1** per operator (CPU vs CPU, random weights) + property tests
      (NaN-free, mask row-sums, RoPE norm-preservation)

## Phase 1c — Attention

- [ ] `MultiHeadAttention` — no-bias projections, QK-RMSNorm on head_dim=80
      (**eps=1e-5**, R5), order: proj → RoPE → QKnorm → PerDimScale →
      sdpa(**scale=√80**, R1); variate-attn = same class, non-causal, no RoPE
- [ ] Gate **A1** on MHA (seq + var configs), CPU vs CPU

## Phase 2 — Transformer stack

- [ ] `MixingTransformer`: pre_ln → attn → `post_ln(out) + x` (exact residual form),
      seq → variate → FFN(relu); reshape `(b,v,n,d) ↔ (b*v,n,d)` / `(b*n,v,d)`
- [ ] `StackedMixingTransformer` ×20
- [ ] KV-cache path (`DecodeCache`): replace `.item()` sync with `mx.eval` /
      `at[]` updates (R4) — can defer to Phase 4 if it blocks progress
- [ ] Gate **A1** at stack level (1 layer + full stack, random weights, real
      weight shapes); self-consistency full-seq vs cache path

## Phase 3 — Model body

- [ ] `util.py`: `revin`, `get_running_stats`/`update_running_stats` (python loop
      over patches is fine first; cumsum closed-form later as optimization),
      `get_output_patch_via_roll` + wrap mask, `stitch_patches`
- [ ] `dense.py`: `ResidualBlock` — **prenorm="none"**, identity_skip=false ⇒
      residual_layer present (gotcha: do not add RMSNorm here); constructor takes
      `input_dim=192` eagerly (no `set_input_dims` lazy rebuild)
- [ ] `model.py`: `_preprocess` → effective leading-only mask (cumprod dim=2) →
      stack → head → `cpm_revin_refine` → inverse RevIN → clip → logits reshape
- [ ] `cpm_revin_refine.py` port (149 LoC)
- [ ] Gate **A1** on full `forward()` (random + real weights, aux outputs compared)

## Phase 4 — Decode / forecasting API

- [ ] `decode()`: ctx/horizon padding, stitching forecast-patch math, linear
      detrending incl. covariate groups + trend re-add, horizon CPM mask
- [ ] `mlx_timesfm.load(path)` + `model.forecast(...)` public API (F5)
- [ ] Gate **A2**: end-to-end fixtures (5 series × h∈{32,128,512}) within tolerance
- [ ] (P2) sklearn-ish `TimesFM3Forecaster` wrapper

## Phase 5 — End-to-end & precision

- [ ] Real-weights e2e parity run → report committed to `.agents/parity-reports/`
- [ ] fp16/bf16 experiment: record quantile drift vs fp32 (median focus); decide
      whether to expose dtype opt-in (F6)
- [ ] Gate **A2** sign-off

## Phase 6 — Polish

- [ ] `mx.compile` on forward/decode; measure before/after; **then re-run the
      full A1+A2 suite** — compile must not move results; if it does, treat as bug
- [ ] Benchmark: 1000 series ctx512 h128 → `.agents/benchmarks/` (Gate **A3**)
- [ ] README (install / usage / parity & perf tables) incl. **license section**:
      code Apache-2.0, pretrained weights under separate Google terms license
- [ ] `LICENSE` file (Apache-2.0) at repo root
- [ ] Perf candidates: fuse variate-attn permutes; cumsum closed-form running stats
- [ ] Gate **A4**: clean-clone `uv run pytest` green, torch-free

## Definition of done

SPEC A1–A4 all green, README merged, `.venv-torch/` never referenced by package code.
