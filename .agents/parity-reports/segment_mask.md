# Parity report: `segment_mask`

- generated: 2026-09-01 22:59:58
- seed: 0
- devices: torch=cpu, mlx=cpu (SPEC A1)
- criterion: bit-exact
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
| `mask` | (3, 1, 10, 10) | True | 0 |
