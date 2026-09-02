# Parity report: `mixing_layer`

- generated: 2026-09-02 00:26:57
- seed: 0
- devices: torch=cpu, mlx=cpu (SPEC A1)
- criterion: max abs diff <= 0.0001
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
| `mask` | (3, 1, 8, 8) | True | 0 |
| `y` | (1, 3, 8, 1280) | False | 2.52724e-05 |
