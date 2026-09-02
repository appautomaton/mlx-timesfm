# Parity report: `rmsnorm`

- generated: 2026-09-02 00:26:50
- seed: 0
- devices: torch=cpu, mlx=cpu (SPEC A1)
- criterion: max abs diff <= 1e-05
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
| `y` | (4, 300, 1280) | False | 9.53674e-07 |
