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
- **RMSNorm: explicit `eps` everywhere (port config knob)** — never rely on
  framework defaults (R5); match the eps the torch reference env actually uses,
  probed via `torch.nn.RMSNorm(80).eps` and printed in every parity report.
- Do NOT port torch's lazy `ResidualBlock.set_input_dims` hack — take
  `input_dim=2*(ip+op)=192` in the constructor.

## Phase 0 — Scaffolding ✅ done

- [x] `uv init --lib --vcs git`, Python 3.13, `mlx==0.32.2` (+mlx-metal), src-layout
- [x] Dependency red line recorded (N1) — torch only ever in `.venv-torch/`
- [x] `.gitignore` (.venv*, .references/, models symlink), initial commit

## Phase 1a — Parity harness + config loader

Goal: infrastructure to prove every operator exists before any operator is ported.

- [x] `.venv-torch/`: `uv venv .venv-torch --python 3.13` — NOTE: uv-created venvs
      have **no pip inside**; install via
      `uv pip install --python .venv-torch/bin/python -e ".references/timesfm[torch]"`
      (torch comes via the `[torch]` extra — explicit, not by luck; also pulls
      huggingface_hub/safetensors/numpy, fine inside this isolated env;
      `import timesfm3` requires torch and gets it here)
- [x] `uv add --dev pytest ruff` (+numpy for the npz bridge; main .venv dev deps —
      torch is never one of them; mlx 0.32 no longer bundles numpy, so the dev
      group must carry it)
- [x] `config.py`: parse `models/timesfm_3_0/original/config.json` → dataclasses
      mirroring `.references` `configs.py` semantics; single source of truth for
      model construction, tests, and `mlx_timesfm.load()`
- [x] `tests/parity/bridge.py`: a seeded generator writes **inputs AND model
      weights** to npz so both stacks start bit-identical; torch runner + mlx
      runner each emit outputs; comparator diffs into
      `.agents/parity-reports/*.md`. npz artifacts → `tests/parity/artifacts/`
      (gitignored). Both runners **force CPU** (A1). Setup probes
      `torch.nn.RMSNorm(80).eps` + torch/mlx versions into every report header (R5).
- [x] A2 fixture generator: 5 deterministic series (SPEC A2 list) →
      `tests/fixtures/*.csv` + `tests/fixtures/generate.py` (committed; not npz)
- [x] pytest marker `parity` — auto-skip when `.venv-torch/` is absent (A4);
      match via `get_closest_marker` — the *directory* name `tests/parity/` is
      also a pytest keyword, so `"parity" in item.keywords` over-matches
- [x] Smoke: bridge round-trips one trivial op — `mx.maximum(x, 0)` vs
      `torch.nn.functional.relu(x)` → **bit-exact**, report in parity-reports/
- [x] R5 probe result (torch 2.13): `torch.nn.RMSNorm(80).eps` → **None**, not a
      number. Settled at 1b: instantiate torch RMSNorm with an explicit eps and
      diff against mlx at the same eps; record the effective eps in reports.

### 1a review round 1 (all landed)

- [x] **#1 Two-process bridge is hard law**: torch side runs *only* as a
      `.venv-torch/bin/python` subprocess; test code must never `import torch`
      (uv-run pytest has no torch — hardcoding it breaks A4/N1 on day one).
      Locked by a `sys.modules` guard test.
- [x] **#2 Clean clone has no `models/`** (gitignored symlink): tests must not
      default to `load_config("models/…")`. Committed tiny
      `tests/fixtures/config.json` (synthetic, checkpoint-shaped) is the parse
      test path; real-checkpoint test stays skipif.
- [x] **#3 eps policy for the user API**: `rmsnorm_eps` is *not* in
      checkpoint config.json, so a strict R5 reading would make
      `mlx_timesfm.load()` demand torch probing. Decision: the **inference
      boundary** (`load_config`, later `load()`) hard-wires the documented
      port constant `INFERENCE_RMSNORM_EPS = 1e-5`; **parity overrides** it
      with the probed reference value; `from_dict` stays faithful (None=unset).
- [x] **#4 `to_dict()` is checkpoint-shaped**: port-only knobs are stripped on
      export so it round-trips against the official config.json byte-for-key
      (`test_to_dict_is_checkpoint_shaped` locks it).

## Phase 1b — Pointwise operators

- [x] `normalization.py`: `PerDimScale` (init ⇒ scale ≈ 1/√d)
- [x] `transformer.py`: `rope()` — half-rotation style, 3D/4D inputs, arbitrary
      `(b,n)` int positions (R3); `make_attn_mask` / `make_segment_mask` →
      additive −1e9 float masks (R2)
- [x] Gate **A1** per operator (CPU vs CPU, random weights) + property tests
      (NaN-free, mask row-sums, RoPE norm-preservation)
- [x] RoPE parity positions capped at 0..299: cross-framework `powf` for the
      timescale differs ~ulp and the phase error amplifies ≈ |pos|·1e-7, so
      pos~1000 sits at the 1e-4 A1 edge with identical math; the gate tests the
      per-batch-position mechanism (R3), long-range behaviour is A2's job.

## Phase 1c — Attention

- [x] `MultiHeadAttention` — no-bias projections, QK-RMSNorm on head_dim=80
      (explicit eps, R5), order: proj → RoPE → QKnorm → PerDimScale →
      sdpa(**scale=√80**, R1); variate-attn = same class, non-causal, no RoPE
- [x] Gate **A1** on MHA (seq + var configs), CPU vs CPU
- [x] Attention kernel decision: **manual path is the port default** —
      `mx.fast.scaled_dot_product_attention` runs head_dim=80 but deviates
      ~3.1e-3 from manual attention on CPU (measured, mlx 0.32.2), above the
      A1 1e-4 gate. Torch parity also runs the reference manual branch
      (`use_sdpa=False`, `rescale_logits=False`) → identical math, net logit
      scale √head_dim preserved (Q pre-multiplied, R1). Fast kernel becomes a
      Phase-6 performance option under A2 tolerances.

## Phase 2 — Transformer stack

- [x] `MixingTransformer`: pre_ln → attn → `post_ln(out) + x` (exact residual form),
      seq → variate → FFN(relu); reshape `(b,v,n,d) ↔ (b*v,n,d)` / `(b*n,v,d)`
- [x] `StackedMixingTransformer` ×20 (key/shape tree asserted equal to the
      reference state dict: `layers.{i}.…`, 228 keys, bit-exact)
- [ ] **DEFERRED (re-deferred at Phase 4 close, 2026-09-02)**: KV-cache path
      (`DecodeCache`): `decode()` landed WITHOUT a cache (reference decode is
      full-forward per call too), so this is now purely a Phase-6 perf item,
      not a correctness dependency. Replace `.item()` sync with `mx.eval` /
      `at[]` updates (R4); self-consistency check (full-seq vs cache) gates it.
- [x] Gate **A1** at stack level (1 layer real dims + ×20 small-dim stack,
      random weights, real weight shapes)

## Phase 3 — Model body

- [x] `util.py`: `revin`, `get_running_stats`/`update_running_stats` (python loop
      over patches is fine first; cumsum closed-form later as optimization),
      `get_output_patch_via_roll` + wrap mask, `stitch_patches`
- [x] `dense.py`: `ResidualBlock` — **prenorm="none"**, identity_skip=false ⇒
      residual_layer present (gotcha: do not add RMSNorm here); constructor takes
      `input_dim` eagerly (192 at real dims; no `set_input_dims` lazy rebuild)
- [x] `model.py`: `_preprocess` → effective leading-only mask (cumprod dim=2) →
      stack → head → `cpm_revin_refine` → inverse RevIN → clip → logits reshape
      (reference fidelity: running stats computed BEFORE CPM mask lands in
      `masks`; `outputs["revin_stats"]` is the PRE-refinement tuple)
- [x] `cpm_revin_refine.py` port (149 LoC)
- [x] Gate **A1** on full `forward()` — random weights, ≥2 seeds, aux outputs
      compared (logits 3–5e-7; revin stats + preprocessing bit-exact; seq
      masks bit-exact). Two cases: `model_forward_small` (CPM path) and
      `model_forward_freeze` (freeze_after branch).
- [x] **Real-weights forward e2e (reviewer item, done 2026-09-02)** — case
      `model_forward_real`: b=2,v=2, 20 patches of 32, CPM on, aux on; torch
      side runs the SAME manual branch (`use_sdpa=False`) with eps force-set
      to PARITY_EPS on both stacks (the labelled A2-debt configuration —
      explicitly NOT official-default-kernel parity). Report
      `.agents/parity-reports/model_forward_real.md`: **PASS** — logits
      2.7e-5, 20 seq masks bit-exact, revin stats ≤1e-6;
      `aux_transformer_output` 2.44e-4 measured against a scale-justified
      1e-3 gate (|x|≈223 ⇒ 16 fp32 ulps accumulated over 20 layers; the
      final logits, which are what the gate cares about, hold plain 1e-4).

### Phase 2 review round 1 — A2 debts (land these when wiring real weights)

- [x] **A2 must not claim "official default kernel parity"** — landed option
      (a) 2026-09-02: every real-weights run (forward parity + A2 e2e) runs
      torch with `use_sdpa=False` and is *labelled* as manual-branch
      comparison in its report; no official-SDPA parity claim is made
      anywhere. (b) stays open as an optional extra measurement.
- [x] **RMSNorm eps mismatch** — landed 2026-09-02: all real-weights runs
      force `PARITY_EPS=1e-5` on BOTH stacks (torch via `_force_eps` walk,
      mlx via `load(rmsnorm_eps=…)`); the probe value of torch 2.13's default
      is recorded in each report's confounders table instead of assumed.
- [ ] RoPE long-context positions (>299) — **still open after A2**. Measured
      coverage at A2 e2e: largest cell (ar1_mv3, ctx 1024, h 512) reaches
      patch position 47 only; the 0..299 parity sweep (line above, Phase 1b)
      remains the furthest verified. A >299 case needs ≥~10k-point context —
      out of scope for the A2 fixtures; keep as a named debt (would show as
      horizon-growing systematic if broken; h512 slacks track h32 slacks).

## Phase 4 — Decode / forecasting API

- [x] `decode()`: ctx/horizon padding, stitching forecast-patch math, linear
      detrending incl. covariate groups + trend re-add, horizon CPM mask
      (2026-09-02; torch-faithful, incl. the "pad values with 0 / masks with
      True" asymmetry; validated CPU-vs-CPU inside 3e-6 via forward-spy)
- [x] `mlx_timesfm.load(path)` + `model.forecast(...)` public API (F5)
      — strict `load_parameters` (R6), no converted artifact; also
      `__call__ = forward` class alias on TimesFM3 (MLX dispatch quirk:
      special methods are never looked up on instances, so the alias must be
      a class attribute). Checkpoint key-tree locked by skipif test:
      445 tensors / 330,710,976 params, strict-load rejects partial dicts.
- [x] Gate **A2**: end-to-end fixtures, 5 series × h∈{32,128,512} = 15 cells.
      **Decision (review round): split the gate by device.**
      * `test_a2_end_to_end_fixtures_cpu` — torch-CPU vs mlx-CPU, HARD gate,
        all 15 cells PASS, worst slack 0.005 (also inside the tighter
        5e-4·σ informational band SPEC contemplates for CPU runs).
        Report `a2_e2e.md`.
      * `test_a2_end_to_end_fixtures_gpu` — literal torch-CPU vs mlx-GPU
        crossing, all 15 cells measured in `a2_e2e_gpu.md`; 9 cells at slack
        1.09–6.05 ⇒ documented **xfail, conditioned on a live matmul probe**
        (fails hard on any accurate-matmul GPU; flips to plain PASS when MLX
        exposes fp32 precision control). `beyond_band` crossings = 0 in every
        GPU cell — band excess without any ordering deviation.
      * near-flat decision: per-fixture (per-target-series) σ was already the
        A2 rule — near_flat σ≈0.79 is not degenerate; no global-tolerance
        fallback needed, nothing special-cased.
- [ ] (P2) sklearn-ish `TimesFM3Forecaster` wrapper

### Phase 4 review round — environment finding (2026-09-02)

- **MLX 0.32.2 fp32 `matmul` on Apple M5 Max runs a reduced-precision
  (tf32-like) tensor path**: single-matmul rel err vs fp64 reference ≈
  7.4e-4–8.0e-4 for m≥2 across shapes, deterministic; MLX CPU ≈ 1e-6;
  m=1 (gemv) full precision; **no precision knob exists in 0.32.2**
  (`dir(mx)` / `mx.metal` expose none). This is ~2 orders of magnitude
  above what A2's 2e-3·σ was written to absorb and is NOT port-code: same
  decode inputs agree CPU-vs-CPU at 3e-6, and GPU drift is visible already
  at the first ResidualBlock output (rel 7.6e-4).
  **Debt: re-run the GPU cells when MLX gains an fp32 precision control
  (or on non-M5 hardware); the test automates this via the probe.**

## Phase 5 — End-to-end & precision

- [x] Real-weights e2e parity run → reports committed: `model_forward_real.md`
      (forward, PASS), `a2_e2e.md` (CPU gate, PASS), `a2_e2e_gpu.md`
      (GPU crossing, measured + conditioned xfail — see Phase 4 finding)
- [ ] fp16/bf16 experiment: record quantile drift vs fp32 (median focus); decide
      whether to expose dtype opt-in (F6). Note: if M5's fp32 matmul already
      emulates tf32-precision, fp16/bf16 loss may be smaller than expected —
      measure, don't assume.
- [x] Gate **A2** sign-off (2026-09-02, with one recorded environment debt):
      numerics gate green CPU-vs-CPU (worst slack 0.005, max |diff| 7.5e-6,
      zero beyond-band crossings); GPU crossing 6/15 in band, 9/15 xfailed
      solely to the measured M5 reduced-precision fp32 matmul (probe-gated,
      auto-tightens). No official-default-kernel claim anywhere (SDPA-off +
      forced-equal eps, labelled in every report).

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
