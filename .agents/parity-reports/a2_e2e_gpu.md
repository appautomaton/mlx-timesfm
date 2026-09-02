# Parity report: `a2_e2e_gpu` (A2 fixtures × horizons)

- generated: 2026-09-02 00:27:33
- gate: **SPEC A2** — per-quantile max abs diff ≤ 2e-3·σ; no new
  quantile crossings beyond the tolerance band (rule below)
- σ = population std of each target series' unmasked context values
- slack = worst |diff| / (2e-3·σ); outside the band if > 1
- **verdict: OUT OF BAND (see notes)**

## Environment / confounders removed (both sides)

| key | value |
|---|---|
| torch version | 2.13.0 |
| mlx version | 0.32.2 |
| torch RMSNorm eps default (probe, R5) | None |
| effective RMSNorm eps (both sides) | 1e-05 |
| attention path (torch) | reference manual (use_sdpa=False, rescale_logits=False) |
| devices | torch=cpu, mlx=gpu |

The official checkpoint ships `use_sdpa: true`; this run force-disables
SDPA on the torch side and equalises RMSNorm eps, so BOTH stacks execute
identical math — this is NOT a claim of official-default-kernel parity
(PLAN Phase 2 review note; the measured-vs-official-SDPA delta remains
a separate A2 debt if ever needed).

Context = first 512 fixture points (1024 for ar1_mv3); horizons
32/128/512; b=1; univariate fixtures v=1, ar1_mv3 v=3; no covariates.

- If cells exceed 2e-3·σ here while the CPU run is clean, the cause is the **GPU fp32 matmul path**, not port math: measured one-matmul rel err vs fp64 reference on this device = 7.6e-04 (true fp32 kernels: ~1e-6; Apple M5-family tensor emulation: ~8e-4, deterministic; no precision knob in mlx 0.32.2). Re-measure when MLX exposes fp32 precision control (PLAN Phase 4).

## Cells

| fixture | h | σ_min | tol 2e-3σ | worst \|diff\| | slack | new x | beyond | ok |
|---|---|---|---|---|---|---|---|---|
| ar1_mv3 | 32 | 0.1409 | 0.000282 | 0.000416 | 1.439 | 0 | 0 | False |
| ar1_mv3 | 128 | 0.1409 | 0.000282 | 0.000416 | 1.439 | 0 | 0 | False |
| ar1_mv3 | 512 | 0.1409 | 0.000282 | 0.000416 | 1.439 | 0 | 0 | False |
| near_flat | 32 | 0.7929 | 0.00159 | 0.000858 | 0.541 | 4 | 0 | True |
| near_flat | 128 | 0.7929 | 0.00159 | 0.0038 | 2.397 | 6 | 0 | False |
| near_flat | 512 | 0.7929 | 0.00159 | 0.00959 | 6.050 | 15 | 0 | False |
| sine | 32 | 0.7076 | 0.00142 | 0.00159 | 1.126 | 4 | 0 | False |
| sine | 128 | 0.7076 | 0.00142 | 0.00182 | 1.287 | 5 | 0 | False |
| sine | 512 | 0.7076 | 0.00142 | 0.00244 | 1.726 | 16 | 0 | False |
| trend | 32 | 1.487 | 0.00297 | 0.000143 | 0.048 | 0 | 0 | True |
| trend | 128 | 1.487 | 0.00297 | 0.000206 | 0.069 | 0 | 0 | True |
| trend | 512 | 1.487 | 0.00297 | 0.000219 | 0.074 | 0 | 0 | True |
| white_noise | 32 | 0.9946 | 0.00199 | 0.00185 | 0.928 | 0 | 0 | True |
| white_noise | 128 | 0.9946 | 0.00199 | 0.00198 | 0.994 | 0 | 0 | True |
| white_noise | 512 | 0.9946 | 0.00199 | 0.00217 | 1.091 | 0 | 0 | False |

New-crossing rule: port inversion (p[j+1] < p[j]) where torch was
ordered (p[j+1] ≥ p[j]). `beyond band` additionally requires the torch
gap > 4e-3·σ, i.e. the inversion cannot be explained by the agreed
per-quantile tolerance — only genuine ordering deviation. Near-tied
quantile inversions INSIDE the band are floating-point noise, not new
structure (they vanish under any elementwise tolerance).
