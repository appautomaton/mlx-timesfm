# Real-data parity matrix

- generated: 2026-09-02 11:07:09
- fixture: UCI Appliances Energy Prediction, final 640 observations
- shape: context=512, horizon=128, fp32
- targets: Appliances, T1, RH_1
- past-only covariates: T2, RH_2
- past/future covariates: T_out, Press_mm_hg
- comparison tolerance: per-target max abs diff <= 2e-3 * context sigma

The aligned profile uses manual attention and RMSNorm eps=1.1920929e-07. The
official profile uses the checkpoint's SDPA path and Torch's dtype-derived
FP32 RMSNorm epsilon (1.1920929e-07); MLX receives that value explicitly.

## Parity comparisons (target outputs only)

| reference | candidate | max abs | RMSE | slack | new x | beyond | pass |
|---|---|---:|---:|---:|---:|---:|---|
| torch_cpu_aligned | torch_mps_aligned | 0.00328064 | 0.000260442 | 0.033 | 0 | 0 | True |
| torch_cpu_aligned | mlx_cpu | 0.003479 | 0.000290102 | 0.035 | 0 | 0 | True |
| torch_cpu_aligned | mlx_gpu | 0.0088501 | 0.00075223 | 0.084 | 0 | 0 | True |
| torch_cpu_official | torch_mps_official | 0.00308228 | 0.000272863 | 0.033 | 0 | 0 | True |
| torch_cpu_official | mlx_cpu | 0.00308228 | 0.000276503 | 0.033 | 0 | 0 | True |
| torch_cpu_official | mlx_gpu | 0.0083313 | 0.000733247 | 0.082 | 0 | 0 | True |
| torch_mps_official | mlx_gpu | 0.0057373 | 0.00047711 | 0.051 | 0 | 0 | True |
| torch_cpu_official | torch_cpu_aligned | 0.000549316 | 5.94753e-05 | 0.004 | 0 | 0 | True |

## Held-out median forecast MAE (diagnostic, not a quality gate)

| backend/profile | Appliances | T1 | RH_1 |
|---|---:|---:|---:|
| torch_cpu_aligned | 60.7202 | 0.341792 | 1.0432 |
| torch_mps_aligned | 60.7202 | 0.341795 | 1.0432 |
| torch_cpu_official | 60.7202 | 0.341792 | 1.0432 |
| torch_mps_official | 60.7202 | 0.341794 | 1.0432 |
| mlx_cpu | 60.7202 | 0.341795 | 1.0432 |
| mlx_gpu | 60.7202 | 0.3418 | 1.04321 |

## Runtime and environment

| backend/profile | seconds |
|---|---:|
| torch_cpu_aligned | 0.1488 |
| torch_cpu_official | 0.1291 |
| torch_mps_aligned | 0.1976 |
| torch_mps_official | 0.0363 |
| mlx_cpu | 0.1570 |
| mlx_gpu | 0.0572 |

- Torch version: 2.13.0
- MLX version: 0.32.2
- MLX_ENABLE_TF32: 0
- PYTORCH_ENABLE_MPS_FALLBACK: <unset>
- output variate order: Appliances, T1, RH_1, T2, RH_2, T_out, Press_mm_hg
- provenance: `tests/fixtures/real/README.md`
- intermediate matrix logits: temporary only
- versioned target-only golden: `tests/fixtures/real/`
