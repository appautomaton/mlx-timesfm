# Third-party notices

## Google Research TimesFM

This package contains an independent Apple MLX implementation derived from the
Google Research TimesFM source code.

Copyright 2026 Google LLC. The upstream source is licensed under the Apache
License 2.0. The files in this project were modified and reimplemented for MLX.

Upstream repository: https://github.com/google-research/timesfm

The TimesFM 3.0 pretrained checkpoint is **not included** in this package. It is
distributed separately under the TimesFM Non-Commercial License v1.0, which
restricts it to non-commercial, non-production use:
https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE

Installing this package does not download the checkpoint or grant rights to it.

## UCI Appliances Energy Prediction dataset

The source distribution's test fixtures include an attributed excerpt from:

> Candanedo, L. (2017). Appliances Energy Prediction [Dataset]. UCI Machine
> Learning Repository. https://doi.org/10.24432/C5VC8G

The dataset is licensed under Creative Commons Attribution 4.0 International
(CC BY 4.0). Fixture provenance and hashes are recorded in
`tests/fixtures/real/README.md` and `tests/fixtures/real/golden_manifest.json`.
