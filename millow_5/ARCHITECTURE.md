# millow_5 架构

## 数据流(粗体为 millow_5 新增)

```
输入(120 列)
    │
    ▼
KV-Aware Tokenizer (W2 sum-pool, 13 user tokens)
    + user_dense_proj non-KV (1)
    + RankMixer item (2)
    │
    ▼
ns_tokens (B, 16, 64)
    │
    ▼ flatten
(B, 1024)
    │
    ▼  ★★★ millow_5 ★★★
┌─────────────────────────────┐
│ DCN-V2 Cross Network (×2)   │
│ x_{l+1} = x_0 ⊙ Wx_l + x_l │
│ + LayerNorm                  │
└──────────┬──────────────────┘
           │
(B, 1024) crossed
    │
    ▼ reshape
(B, 16, 64)
    │
    ▼
┌──────────────────────────────────┐
│  4 × longer-encoder seq paths    │
│  + Query Generator (8 query)     │
│  + HyFormer Block × 2            │
│  (rank_mixer ffn_only fallback)  │
└──────────┬───────────────────────┘
           │
           ▼
       output_proj → 1 logit → BCE
```

## DCN-V2 公式细节

```
x_0 ∈ ℝ^d    (d = num_ns × d_model = 16 × 64 = 1024)
x_0 = flatten(ns_tokens)

For l = 0, 1, ..., L-1:
   wx_l = W_l @ x_l + b_l            # (d,)
   raw  = x_0 ⊙ wx_l + x_l            # 元素乘 + 残差
   x_{l+1} = LayerNorm(raw)           # 防爆

x_L = output (后面 reshape 回 16×64 token)
```

`x_0 ⊙ wx_l` 实现二阶交叉(`x_l[i] * x_0[j]` 在 `wx_l[i]` 经过 `W_l` 时混入)。L 层堆叠产生最高 L+1 阶的交叉。L=2 时覆盖 3 阶交叉,**对推荐场景已经够**(更高阶通常过拟合)。

## 参数量

| 组件 | 参数 |
|---|---|
| W_l ∈ ℝ^(1024 × 1024) | 1024 × 1024 + 1024 = ~1.05M |
| LayerNorm 1024 | 2048 |
| × 2 layers | **~2.1M** |
| 总 dcn 参数 | 占总参数 (~2.4 亿) 不到 1% |

实际显存影响:DCN 中间张量 `(B=64, 1024)`,fp32 = 256KB,跟 W2 几乎一样。

## 与 W2 KV-Aware 的协作

DCN-V2 在**所有 16 个 NS tokens 间做交叉**:

```
Token 序号 → 内容
0..7    KV-aware tokens(8 对 KV 各 1 个)
8..12   非 KV user_int 的 RankMixer tokens
13      非 KV user_dense(fid 61, 87)
14..15  item RankMixer tokens
```

DCN-V2 把这 16 token × 64 dim = 1024 维一起 flatten,**让任意两个 token 的特征**(包括 KV-aware 和 item)**相互交叉**。

baseline 隐式靠 attention 学这种交叉,DCN-V2 把它**显式数学化**,不需要 attention 慢慢学。

## 兼容性

- ✓ KV-Aware sum-pool(从 millow_4 继承,W2 已验证)
- ✓ longer encoder + Top-K 200(从 millow_4 继承)
- ✓ 完整 EMA(从 millow_4 继承)
- ✓ rank_mixer_mode 自动 fallback ffn_only(KV-Aware 共有的)

## 评估器同步

`evaluation/infer.py` 的 `_FALLBACK_MODEL_CFG` 加了 `use_dcn` 和 `dcn_layers`,所以 train_config.json 里的 `use_dcn=True` 会传给 evaluation 重建的 model,strict load 不会失败。
