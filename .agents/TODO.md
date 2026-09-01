# mlx-timesfm — TimesFM 3.0 → MLX 移植 TODO

> 参考实现: `.references/timesfm/src/timesfm3/`(PyTorch, inference-only)
> 权重: `models/timeseries/timesfm_3_0/original/`(safetensors, F32, 445 keys, 1.2GB)
> 权重 key 与 PyTorch state_dict 完全同名;MLX `nn.Linear` weight 也是 `(out, in)`,
> 预期 **无需转置/改名**,`mx.load` 后 `model.update()` 直接灌入。

## 架构速查(config.json)

- 20 层 MixingTransformer: seq-attn(RoPE, 因果) → variate-attn(无 RoPE, 非因果) → FFN
- model_dims=1280, heads=16 → head_dim=80, RMSNorm  everywhere, qk_norm=rms, v_norm=none
- PerDimScale 学习式 Q 缩放(Flax Pax 风格),**注意注意力 scale = √head_dim**(见坑#1)
- input_patch=32, output_patch=64, rolls=2, 9 分位数, stitching 开
- 预处理: 线性去趋势(阈值 0.5)→ patch 化 → 逐 patch running stats(RevIN)→
  roll 出 future-covariate patch → ResBlock(192→1280) → transformer → head(→576) → 反 RevIN

## Phase 0 — 项目脚手架

- [x] `uv init --lib --vcs git`;第一个依赖 `mlx==0.32.2`(+mlx-metal,PyPI 最新)
- [x] **依赖红线(用户定死)**:项目依赖**永远不含** `torch` / `safetensors`
      (MLX 自己能用 `mx.load` 读 safetensors,不需要 python safetensors 包)。
      数值对齐如需 torch 参考实现,只装在独立的 `.venv-torch/`(已 gitignore),
      不进 pyproject/uv.lock。
- [x] `.gitignore`:忽略 `.venv/`、`.venv-torch/`、`.references/`、`models`(软链)
- [ ] dev 依赖:`pytest`(可进 dev group);torch 仅手动装进 `.venv-torch/`
- [ ] 目录: `src/mlx_timesfm/`(model.py, transformer.py, dense.py, normalization.py, util.py, forecaster.py), `scripts/convert_weights.py`, `tests/`
- [ ] `pyproject.toml` 读 config.json 的 loader(config → dataclass,复用 `.references` 的 configs.py 语义)

## Phase 1 — 基础算子(可单测)

- [ ] `PerDimScale`: x * 1.442695041/√d * softplus(s), 参数名 `per_dim_scale.per_dim_scale`
- [ ] `rope(inputs, position)`: 半旋转式(非 interleaved),支持 3D/4D 输入与任意 position(b,n)
- [ ] attention mask 工具: `make_attn_mask`(前导 masked + 因果 + decode offset)、`make_segment_mask`,输出 additive float mask(-1e9)供 sdpa 使用
- [ ] `MultiHeadAttention`: QKV/out proj(无 bias)、head_dim RMSNorm(QK-norm)、PerDimScale、RoPE 顺序 = proj → RoPE → QKnorm → PerDimScale → attn
- [ ] 单测: 与 torch 版逐算子对齐(同随机权重, atol~1e-5)

## Phase 2 — Transformer 栈

- [ ] `MixingTransformer`(seq attn → variate attn → FFN relu, post-LN + residual 的写法严格照抄: `post_ln(attn_out) + x`)
- [ ] `StackedMixingTransformer`(20 层)
- [ ] KV-cache decode 路径(`DecodeCache`, 可先于 Phase 4 缓做)
- [ ] 单测: 随机权重下与 torch 对齐(输入 b,v,n,d 与 patch_mask)

## Phase 3 — 模型主体

- [ ] `ResidualBlock`(prenorm=none 路径!config 里 `prenorm:"none"` → 无 pre_norm; identity_skip=false → residual_layer 192→1280)
- [ ] `revin` / `get_running_stats` / `update_running_stats`(逐 patch python 循环,MLX 里先照抄,`mx.eval` 控制物化)
- [ ] `get_output_patch_via_roll` + wrap_mask、`stitch_patches`
- [ ] `TimesFM3.forward`: _preprocess → effective_patch_mask(cumprod,只挡前导)→ stack → head → cpm_revin_refine → 反 RevIN → clip
- [ ] 单测: 随机权重整模型对齐(`return_aux_outputs` 逐中间量对比)

## Phase 4 — decode / 预测 API

- [ ] `decode()`: padding 到 patch 倍数、线性去趋势(带协变量分组)、horizon CPM mask、stitching 提取、trend 加回
- [ ] 移植 `cpm_revin_refine.py`(149 行,迭代 RevIN 精化)
- [ ] (可选)sklearn 风格 `TimesFM3Forecaster`(参考 timesfm3_forecaster.py,764 行)

## Phase 5 — 权重加载与端到端验证

- [ ] `scripts/convert_weights.py`: 实际上可能只要 `mx.load(safetensors)` + key 过滤(去掉 timesfm 前缀?无外层前缀,预期直通);产出 `models/.../mlx/`
- [ ] 真权重端到端: 同一组随机时序,torch `decode()` vs mlx `decode()`,9 分位逐点对比(目标 max abs diff / rel diff 记录进 .agents/)
- [ ] 精度实验: float32 基线 → float16/bfloat16 误差评估(输出是分位数,关注 median 漂移)

## Phase 6 — 打磨

- [ ] `mx.compile` forward;memory 峰值检查(20 层 b,v,n,d 中间量不小)
- [ ] 简单 benchmark(encode 吞吐 ms/series)+ README(安装/用法/对齐误差表)
- [ ] 性能优化候选: variate-attn 的 permute 融合、running-stats 循环改 cumsum 闭式解(需验证等价性)

## 已知坑

1. **注意力缩放**: Flax MEA(rescale_logits=False)语义 → SDPA 传 `scale=√head_dim=√80`,因为 Q 已被 PerDimScale 类"预缩放"逻辑取代 1/√d。MLX `mx.fast.scaled_dot_product_attention` 默认 scale=1/√d,必须显式覆盖。**最容易对齐翻车点。**
2. mask 语义: 参考实现 True=attend;softmax 前加 -1e9。MLX sdpa 的 mask 是 additive,方向别搞反。
3. `effective_patch_mask = cumprod(patch_mask)`——只遮前导 padding patch,horizon 的全 mask patch 仍可见。
4. RoPE position 是 int32 (b,n) 且 decode 时要加 next_index 偏移;MLX `nn.RoPE` 内置版不支持任意 position 输入,需要自己写。
5. MLX 惰性求值: `torch.where(det==0,...)` 一类分支没有 eager 副作用问题,但 `next_index[0].item()` 这种同步点在 KV-cache 路径要换成 `mx.eval` 或改成图内索引。
6. ResidualBlock 的 prenorm 是 **"none"**(见 config.json)——不要顺手加 RMSNorm。
7. QK-norm 作用在 **head_dim=80** 上且带可学 weight(MLX `nn.RMSNorm(d)` 默认 affine=True,匹配)。

## 不做

- 训练路径、jax/flax 版、v1 归档、evaluator/benchmarks
