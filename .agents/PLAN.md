# PLAN — mlx-timesfm

> Current implementation and release tracker. Requirements and acceptance gates
> are defined in `SPEC.md`; frozen oracle provenance is in
> `tests/fixtures/real/golden_manifest.json`.

## Completed foundation

- [x] Native MLX TimesFM 3.0 model, transformer stack, RevIN/CPM refinement,
      strict checkpoint loading, `decode()`, and `forecast()`.
- [x] Original fp32 checkpoint structure validated: 445 tensors and
      330,710,976 parameters loaded directly with `mx.load()`.
- [x] Full-fp32 policy established with `MLX_ENABLE_TF32=0` by default.
- [x] Effective fp32 RMSNorm epsilon pinned to `1.1920928955078125e-7`.
- [x] Live reference parity completed and retired; no active project dependency,
      import, environment, subprocess, or test uses PyTorch.
- [x] Real multivariate/covariate golden frozen from the verified oracle with
      source, reference revision, checkpoint/config, and output hashes.
- [x] MLX CPU/GPU golden regression and quantile-ordering gates pass.
- [x] Clean MLX-only clone test passes; checkpoint-dependent tests skip cleanly.

## Initial PyPI release readiness

- [x] Reserve pending PyPI trusted publisher: package `mlx-timesfm`, owner
      `appautomaton`, repository `mlx-timesfm`, workflow `workflow.yml`,
      environment `pypi`.
- [x] Remove retired parity runners, reports, temporary artifacts, and unused
      synthetic fixtures; harden `.gitignore` for local-only assets.
- [x] Complete PEP 621/639 metadata, Apache-2.0 licensing, third-party notices,
      version exposure, changelog, and public README with focused badges.
- [x] Add CI and release-triggered trusted-publishing workflows with current
      actions pinned to immutable commit SHAs and minimal permissions.
- [x] Add release checks for tag/version agreement, repository hygiene,
      distribution metadata/contents, and installed-package imports.
- [x] Run `uv sync --locked`, lint, tests, and `uv build --no-sources`.
- [x] Inspect wheel/sdist metadata and contents; install and smoke-test both in
      isolated environments.
- [x] Create/configure `appautomaton/mlx-timesfm`, its `pypi` environment, branch
      protections, and release settings; do not publish before a validated tag.

## Post-release engineering

- [ ] Benchmark 1000 univariate series (context 512, horizon 128) against the
      A3 target (<60 seconds on GPU); record under `.agents/benchmarks/`.
- [ ] Measure `mx.compile` on forward/decode and re-run the golden suite.
- [ ] Evaluate fp16/bf16 and explicit reduced-precision fp32 as opt-in modes.
- [ ] Validate RoPE positions above 299 with a long-context golden.
- [ ] Consider the optional sklearn-style forecaster wrapper.
- [ ] Consider performance-only KV caching and running-stat/attention fusions.

## Release definition

The release is ready when metadata, documentation, licensing, CI, trusted
publishing, artifact inspection, isolated installation, tests, and lint are all
green, with no model weights or local development state in either distribution.
