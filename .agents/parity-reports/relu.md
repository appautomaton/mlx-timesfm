# Parity report: `relu`

- generated: 2026-09-01 22:17:21
- seed: 0
- devices: torch=cpu, mlx=cpu (SPEC A1)
- criterion: bit-exact
- **verdict: PASS**

## Environment probe (SPEC R5)

| stack | version | rmsnorm eps default | device |
|---|---|---|---|
| torch | 2.13.0 | None | cpu |
| mlx | 0.32.2 | n/a | cpu |

## Outputs

| output | shape | bit-exact | max abs diff |
|---|---|---|---|
| `y` | (4096,) | True | 0 |
