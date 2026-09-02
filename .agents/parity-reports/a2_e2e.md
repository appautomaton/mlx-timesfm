# Parity report: `a2_e2e` (A2 fixtures × horizons)

- generated: 2026-09-02 00:27:31
- gate: **SPEC A2** — per-quantile max abs diff ≤ 2e-3·σ; no new
  quantile crossings beyond the tolerance band (rule below)
- σ = population std of each target series' unmasked context values
- slack = worst |diff| / (2e-3·σ); outside the band if > 1
- **verdict: PASS**

## Environment / confounders removed (both sides)

| key | value |
|---|---|
| torch version | 2.13.0 |
| mlx version | 0.32.2 |
| torch RMSNorm eps default (probe, R5) | None |
| effective RMSNorm eps (both sides) | 1e-05 |
| attention path (torch) | reference manual (use_sdpa=False, rescale_logits=False) |
| devices | torch=cpu, mlx=cpu |

The official checkpoint ships `use_sdpa: true`; this run force-disables
SDPA on the torch side and equalises RMSNorm eps, so BOTH stacks execute
identical math — this is NOT a claim of official-default-kernel parity
(PLAN Phase 2 review note; the measured-vs-official-SDPA delta remains
a separate A2 debt if ever needed).

Context = first 512 fixture points (1024 for ar1_mv3); horizons
32/128/512; b=1; univariate fixtures v=1, ar1_mv3 v=3; no covariates.

- Devices: torch=cpu vs mlx=cpu — the numerics gate (SPEC A2 contemplates this tighter, kernel-neutral reading; every cell here also sits far inside the 5e-4·σ band).
- The literal torch-CPU vs mlx-GPU crossing is recorded separately in `a2_e2e_gpu.md` (reduced-precision GPU fp32 matmul on Apple M5 — see that report).

## Cells

| fixture | h | σ_min | tol 2e-3σ | worst \|diff\| | slack | new x | beyond | ok |
|---|---|---|---|---|---|---|---|---|
| ar1_mv3 | 32 | 0.1409 | 0.000282 | 5.66e-07 | 0.002 | 0 | 0 | True |
| ar1_mv3 | 128 | 0.1409 | 0.000282 | 7.75e-07 | 0.003 | 0 | 0 | True |
| ar1_mv3 | 512 | 0.1409 | 0.000282 | 7.75e-07 | 0.003 | 0 | 0 | True |
| near_flat | 32 | 0.7929 | 0.00159 | 1.43e-06 | 0.001 | 0 | 0 | True |
| near_flat | 128 | 0.7929 | 0.00159 | 4.89e-06 | 0.003 | 0 | 0 | True |
| near_flat | 512 | 0.7929 | 0.00159 | 7.51e-06 | 0.005 | 1 | 0 | True |
| sine | 32 | 0.7076 | 0.00142 | 2.03e-06 | 0.001 | 0 | 0 | True |
| sine | 128 | 0.7076 | 0.00142 | 2.86e-06 | 0.002 | 0 | 0 | True |
| sine | 512 | 0.7076 | 0.00142 | 2.44e-06 | 0.002 | 0 | 0 | True |
| trend | 32 | 1.487 | 0.00297 | 3.34e-06 | 0.001 | 0 | 0 | True |
| trend | 128 | 1.487 | 0.00297 | 4.77e-06 | 0.002 | 0 | 0 | True |
| trend | 512 | 1.487 | 0.00297 | 8.58e-06 | 0.003 | 0 | 0 | True |
| white_noise | 32 | 0.9946 | 0.00199 | 2.62e-06 | 0.001 | 0 | 0 | True |
| white_noise | 128 | 0.9946 | 0.00199 | 3.1e-06 | 0.002 | 0 | 0 | True |
| white_noise | 512 | 0.9946 | 0.00199 | 2.62e-06 | 0.001 | 0 | 0 | True |

New-crossing rule: port inversion (p[j+1] < p[j]) where torch was
ordered (p[j+1] ≥ p[j]). `beyond band` additionally requires the torch
gap > 4e-3·σ, i.e. the inversion cannot be explained by the agreed
per-quantile tolerance — only genuine ordering deviation. Near-tied
quantile inversions INSIDE the band are floating-point noise, not new
structure (they vanish under any elementwise tolerance).
