# Copyright 2026 Google LLC
# Modifications Copyright 2026 AppAutomaton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Iterative RevIN refinement for CPM-masked patches.

Exact port of ``timesfm3.cpm_revin_refine.cpm_iterative_revin_refine``.
Python loop over patches (the reference loops too — carry state is
sequential). Replaces frozen RevIN stats at CPM-masked patches with stats
that also incorporate model-estimated values for preceding CPM patches.
"""

from __future__ import annotations

import mlx.core as mx

from . import util


def cpm_iterative_revin_refine(
    raw_logits: mx.array,
    revin_n: mx.array,
    revin_mu: mx.array,
    revin_sigma: mx.array,
    patch_cpm_mask: mx.array,
    median_q_idx: int,
    rolls: int,
    patch_len: int,
    num_quantiles: int,
    value_clip: float = 1e9,
) -> tuple[mx.array, mx.array]:
    """Returns (refined_mu, refined_sigma), each (b, v, n).

    Non-CPM positions equal the inputs; CPM positions fold in estimates of
    all preceding CPM patches in the block (and earlier blocks' estimates).
    """
    b, v, n_patches, _ = raw_logits.shape

    # (b,v,n,op*q) -> (b,v,n,rolls,patch_len,q) -> median -> (b,v,n,rolls,pl)
    median_logits = raw_logits.reshape(b, v, n_patches, rolls, patch_len, num_quantiles)[
        ..., median_q_idx
    ]

    zeros_bv = mx.zeros((b, v), dtype=mx.float32)
    carry_n = carry_mu = carry_sigma = zeros_bv
    anchor_predicted_values = mx.zeros((b, v, rolls, patch_len), dtype=mx.float32)
    block_offset = mx.zeros((b,), dtype=mx.int32)

    refined_mu_list, refined_sigma_list = [], []
    step_masks = mx.zeros((b, v, patch_len), dtype=mx.bool_)

    roll_arange = mx.arange(rolls, dtype=mx.int32)[None, :]

    for i in range(n_patches):
        actual_n = revin_n[:, :, i]
        actual_mu = revin_mu[:, :, i]
        actual_sigma = revin_sigma[:, :, i]
        current_step_logits = median_logits[:, :, i]
        is_cpm = patch_cpm_mask[:, i : i + 1]  # (b, 1)

        # pick anchor_predicted_values[..., block_offset[b], ...] per batch
        offset_onehot = (roll_arange == block_offset[:, None]).astype(mx.float32)
        predicted_values_step = mx.einsum("br,bvrp->bvp", offset_onehot, anchor_predicted_values)

        new_n, new_mu, new_sigma = util.update_running_stats(
            carry_n, carry_mu, carry_sigma, predicted_values_step, step_masks
        )

        out_n = mx.where(is_cpm, new_n, actual_n)
        out_mu = mx.where(is_cpm, new_mu, actual_mu)
        out_sigma = mx.where(is_cpm, new_sigma, actual_sigma)

        new_block_offset = mx.where(
            is_cpm.squeeze(-1),
            mx.remainder(block_offset + 1, rolls),
            mx.zeros_like(block_offset),
        )
        should_update_anchor = new_block_offset == 0

        step_predicted_values = util.revin(
            current_step_logits, out_mu, out_sigma, reverse=True
        )
        step_predicted_values = mx.clip(step_predicted_values, -value_clip, value_clip)

        anchor_predicted_values = mx.where(
            should_update_anchor[:, None, None, None],
            step_predicted_values,
            anchor_predicted_values,
        )
        carry_n, carry_mu, carry_sigma = out_n, out_mu, out_sigma
        block_offset = new_block_offset

        refined_mu_list.append(out_mu)
        refined_sigma_list.append(out_sigma)

    return (
        mx.stack(refined_mu_list, axis=2),
        mx.stack(refined_sigma_list, axis=2),
    )
