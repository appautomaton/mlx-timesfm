# Parity report: `model_forward_freeze`

- generated: 2026-09-02 00:26:48
- seed: 1
- devices: torch=cpu, mlx=cpu (SPEC A1)
- criterion: max abs diff <= 0.0001 (per-output overrides: revin_mu: 1e-05, revin_sigma: 1e-05, seq_mask_0: bit-exact, seq_mask_1: bit-exact)
- **verdict: PASS**

## Environment probe (SPEC R5)

| key | value |
|---|---|
| torch version | 2.13.0 |
| mlx version | 0.32.2 |
| torch RMSNorm eps default (probe, R5) | None |
| effective RMSNorm eps (both sides) | 1e-05 |
| attention path | port manual (Q pre-scaled by sqrt(head_dim)) |
| device | cpu |

## Outputs

| output | shape | bit-exact | max abs diff |
|---|---|---|---|
| `aux_resblock_input` | (2, 2, 6, 24) | True | 0 |
| `aux_transformer_input` | (2, 2, 6, 32) | True | 0 |
| `aux_transformer_output` | (2, 2, 6, 32) | False | 4.29153e-06 |
| `revin_mu` | (2, 2, 6) | True | 0 |
| `revin_sigma` | (2, 2, 6) | True | 0 |
| `seq_mask_0` | (4, 1, 6, 6) | True | 0 |
| `seq_mask_1` | (4, 1, 6, 6) | True | 0 |
| `y` | (2, 2, 6, 8, 3) | False | 5.96046e-07 |
