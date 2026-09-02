# Parity report: `model_forward_real`

- generated: 2026-09-02 00:26:50
- seed: 0
- devices: torch=cpu, mlx=cpu (SPEC A1)
- criterion: max abs diff <= 0.0001 (per-output overrides: aux_transformer_output: 0.001, revin_mu: 1e-05, revin_sigma: 1e-05, seq_mask_0: bit-exact, seq_mask_1: bit-exact, seq_mask_10: bit-exact, seq_mask_11: bit-exact, seq_mask_12: bit-exact, seq_mask_13: bit-exact, seq_mask_14: bit-exact, seq_mask_15: bit-exact, seq_mask_16: bit-exact, seq_mask_17: bit-exact, seq_mask_18: bit-exact, seq_mask_19: bit-exact, seq_mask_2: bit-exact, seq_mask_3: bit-exact, seq_mask_4: bit-exact, seq_mask_5: bit-exact, seq_mask_6: bit-exact, seq_mask_7: bit-exact, seq_mask_8: bit-exact, seq_mask_9: bit-exact)
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
| `aux_resblock_input` | (2, 2, 20, 192) | False | 8.34465e-07 |
| `aux_transformer_input` | (2, 2, 20, 1280) | False | 1.52588e-05 |
| `aux_transformer_output` | (2, 2, 20, 1280) | False | 0.000244141 |
| `revin_mu` | (2, 2, 20) | False | 9.53674e-07 |
| `revin_sigma` | (2, 2, 20) | False | 4.76837e-07 |
| `seq_mask_0` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_1` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_10` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_11` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_12` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_13` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_14` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_15` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_16` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_17` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_18` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_19` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_2` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_3` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_4` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_5` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_6` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_7` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_8` | (4, 1, 20, 20) | True | 0 |
| `seq_mask_9` | (4, 1, 20, 20) | True | 0 |
| `y` | (2, 2, 20, 64, 9) | False | 2.70605e-05 |
