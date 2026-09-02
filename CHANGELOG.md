# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-09-02

### Added

- Native MLX implementation of the TimesFM 3.0 inference stack.
- Direct loading of the original `config.json` and `model.safetensors` files.
- Univariate and multivariate forecasting with past-only and past/future
  covariates, quantile outputs, masking, detrending, stitching, and CPM-RevIN.
- Full-fp32 MLX CPU/GPU parity policy and versioned real-data golden regression.
- Strict checkpoint structure validation for 445 tensors and 330.7M parameters.
- MLX-only test suite with no PyTorch or safetensors package dependency.

[0.0.1]: https://github.com/appautomaton/mlx-timesfm/releases/tag/v0.0.1
