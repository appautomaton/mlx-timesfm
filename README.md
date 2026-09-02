<div align="center">

# mlx-timesfm

**Pure MLX inference for Google TimesFM 3.0 on Apple silicon.**

[![PyPI](https://img.shields.io/pypi/v/mlx-timesfm?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-timesfm/)
[![CI](https://github.com/appautomaton/mlx-timesfm/actions/workflows/ci.yml/badge.svg)](https://github.com/appautomaton/mlx-timesfm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-native-000000?style=flat-square&logo=apple&logoColor=white)](https://support.apple.com/mac/)
[![MLX](https://img.shields.io/badge/backend-MLX-7C3AED?style=flat-square)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/code-Apache--2.0-4C8BF5?style=flat-square)](LICENSE)

[**PyPI**](https://pypi.org/project/mlx-timesfm/) ·
[**Source**](https://github.com/appautomaton/mlx-timesfm)

</div>

`mlx-timesfm` is an independent, inference-only implementation of TimesFM 3.0
for Apple MLX. It loads the original 330.7M-parameter FP32 checkpoint directly,
without converted weights or a PyTorch runtime, and runs forecasting on Apple
silicon through MLX and Metal.

> [!IMPORTANT]
> The official TimesFM 3.0 checkpoint is not included. Google distributes it
> under the **TimesFM Non-Commercial License v1.0**, which restricts the model
> to non-commercial, non-production use. Installing this Apache-2.0 package
> does not download the checkpoint or grant rights to it. Read the
> [checkpoint license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
> before downloading or using the weights.

## Features

- Native MLX model, transformer, normalization, masking, and forecast decode.
- Direct `mx.load()` of the original `model.safetensors` checkpoint.
- Univariate and multivariate forecasts with nine quantiles.
- Past-only and past/future dynamic covariates.
- Linear detrending, stitching, contiguous patch masking, and iterative RevIN.
- Strict validation of all 445 checkpoint tensors and their shapes.
- MLX-only CPU/GPU golden regression against a versioned real-data oracle.
- Full-FP32 matrix kernels by default for numerically stable inference.

## Status

| Area | Status |
|---|---|
| Core TimesFM 3.0 inference | Implemented |
| Original checkpoint loading | Validated |
| `decode()` and `forecast()` | Implemented |
| Multivariate targets and covariates | Validated |
| MLX CPU/GPU golden parity | Passing |
| PyTorch-free runtime and tests | Enforced |
| Training and fine-tuning | Out of scope |
| KV-cache and compiled decode optimizations | Not yet implemented |

This is an initial alpha release. The reference inference path is implemented
and parity-tested, but broader dataset validation and performance tuning remain
ongoing.

## Requirements

- Apple silicon Mac with Metal support
- Python 3.13
- MLX 0.32.x
- A separately downloaded TimesFM 3.0 checkpoint

## Installation

Install from PyPI with uv:

```sh
uv add mlx-timesfm
```

Or install the current checkout for development:

```sh
git clone https://github.com/appautomaton/mlx-timesfm.git
cd mlx-timesfm
uv sync --locked
```

## Model weights

Review and accept Google's checkpoint terms, then download the original files.
One option is the Hugging Face CLI in an isolated uv tool environment:

```sh
uvx --from huggingface-hub hf download google/timesfm-3.0-pytorch \
  --local-dir models/timesfm_3_0/original
```

The checkpoint directory must contain:

```text
models/timesfm_3_0/original/
├── config.json
└── model.safetensors
```

Weights remain local and are excluded from Git and all package distributions.

## Quick start

```python
import numpy as np

import mlx_timesfm

model = mlx_timesfm.load("models/timesfm_3_0/original")

# A batch of two univariate series, each with 512 context points.
target = np.random.default_rng(0).normal(size=(2, 512)).astype(np.float32)
quantiles = model.forecast(target, horizon=128)

print(quantiles.shape)  # (2, 1, 128, 9)
median = quantiles[..., 4]
```

Inputs may be one-dimensional (`time`), two-dimensional (`batch, time`), or
three-dimensional (`batch, variate, time`). The output is always
`(batch, variate, horizon, 9)`, ordered from the 0.1 through 0.9 quantiles.

### Covariates

```python
target = np.zeros((1, 3, 512), dtype=np.float32)
past_only = np.zeros((1, 2, 512), dtype=np.float32)
past_future = np.zeros((1, 2, 512 + 128), dtype=np.float32)

quantiles = model.forecast(
    target,
    horizon=128,
    past_only_covariates=past_only,
    past_future_covariates=past_future,
)
```

Boolean masks use `True` for missing or invalid values. `forecast()` accepts a
global context mask; `decode()` exposes separate masks for targets and both
covariate groups.

## Numerical precision

FP32 is the supported correctness baseline. MLX can otherwise choose
reduced-precision matrix kernels while keeping FP32 array dtypes, so importing
`mlx_timesfm` defaults `MLX_ENABLE_TF32` to `0` before MLX computation. An
explicit environment setting is preserved:

```sh
# Optional faster/reduced-precision experiment; not the parity baseline.
MLX_ENABLE_TF32=1 uv run python your_forecast.py
```

The model also pins the effective FP32 RMSNorm epsilon used to establish the
golden oracle. FP16 and BF16 inference are not currently supported profiles.

## Parity

The permanent regression uses 512 real context observations and a 128-step
forecast from the UCI Appliances Energy Prediction dataset. It exercises three
targets, two past-only covariates, two past/future covariates, every forecast
quantile, and quantile ordering.

| Comparison | Maximum absolute difference | Tolerance used | Result |
|---|---:|---:|---|
| Frozen oracle → MLX CPU | 0.00308228 | 3.3% of allowed band | Pass |
| Frozen oracle → MLX GPU | 0.00833130 | 8.2% of allowed band | Pass |

The golden manifest pins the dataset, source revision, checkpoint/config
hashes, execution profile, output shape, and tolerance. These numbers establish
implementation parity; they are not claims about forecast accuracy on every
downstream dataset.

## Development

```sh
uv sync --locked
uv run ruff check .
uv run pytest
uv build --no-sources
```

The normal test suite is MLX-only. With the original checkpoint present it
runs the golden on MLX CPU and GPU; without the checkpoint, only the seven
weight-dependent checks skip. A permanent hygiene test rejects retired
framework imports and dependencies.

## Release process

GitHub releases with tags matching `v<version>` trigger
`.github/workflows/workflow.yml`. The workflow validates tag/version agreement,
tests the project, builds and inspects both distributions, smoke-tests installed
artifacts, and publishes through PyPI Trusted Publishing. Releases are not
published from developer-machine credentials.

The pending PyPI publisher must match:

- Owner: `appautomaton`
- Repository: `mlx-timesfm`
- Workflow: `workflow.yml`
- Environment: `pypi`

## Licensing

Project source code is Apache-2.0 licensed; see [LICENSE](LICENSE). Third-party
attributions are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The separately distributed TimesFM 3.0 checkpoint has different, restrictive
terms described above.

TimesFM is a Google trademark. This independent project is not affiliated with
or endorsed by Google.

---

Built and maintained by [AppAutomaton](https://appautomaton.renocrypt.com).
Explore more MLX-native projects on
[GitHub](https://github.com/appautomaton) and
[Hugging Face](https://huggingface.co/appautomaton).
