"""Pytest setup for the full-fp32 MLX baseline."""

import os

# Enforce the project's full-FP32 test baseline before any collected module can
# launch an MLX kernel. Keep an explicit caller override for precision studies.
os.environ.setdefault("MLX_ENABLE_TF32", "0")
