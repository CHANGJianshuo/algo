# millow_4 架构

## 整体数据流

```
                    输入(120 列 parquet)
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
       user_int(46)   user_dense(10)   item_int(14) + 4 × seq
            │               │                │
            ▼               ▼                ▼
   ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
   │KVAwareUserNS │  │ user_dense_  │  │RankMixerNS  │
   │  (W2 sum-pool│  │   proj       │  │ (item, 2 tok)│
   │   8 KV+5 non)│  │ (non-KV only │  │             │
   │              │  │  fid 61, 87) │  │             │
   └──────┬───────┘  └──────┬───────┘  └──────┬──────┘
          │                 │                 │
       13 tok            1 tok             2 tok
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
                       16 NS tokens (cat)
                            │
       ┌────────────────────┼─────────────────┐
       │                                       │
       ▼                                       ▼
┌──────────────────┐                   ┌─────────────────┐
│seq_a/b/c/d encode │ ← --seq_top_k 200 │  Query Generator │
│ (longer encoder, │   per-domain      │  (NS + seq pool  │
│  top-K compress) │                    │   → 8 query tok)│
│ seq_max_lens     │                    │                  │
│ a/b/c=512, d=1024│                    │                  │
└──────────────────┘                    └────────┬─────────┘
                                                 │
                              8 query + 16 NS = 24 tokens (T=24)
                                                 │
                                                 ▼
                                  ┌──────────────────────┐
                                  │ HyFormer Block × 2   │
                                  │ (rank_mixer ffn_only │
                                  │  自动 fallback)       │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                       8 query tokens
                                             │
                                             ▼
                                       output_proj → 1 logit
                                             │
                                             ▼
                                          BCE loss
                                             │
                                             ▼
                                  Adagrad(emb) + AdamW(dense)
                                  + EMA shadow update
                                             │
                                       eval: EMA swap-in
```

## KV-Aware sum-pool(回到 W2)

```python
# 每对 KV(共 8 对):
k_emb = Embedding[k_vocab](K_ids)            # (B, N, emb_dim), N=3~10
mask = (K_ids != 0).float()                   # (B, N)
v_masked = V_vals * mask                      # (B, N)

weighted = (k_emb * v_masked.unsqueeze(-1)).sum(dim=1)   # (B, emb_dim)
v_norm = v_masked.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
pooled = weighted / v_norm                                # (B, emb_dim)

token = SiLU(Linear(emb_dim, d_model)(pooled)).unsqueeze(1)
```

**关键点**:
- 没有可学习的 attention 权重
- V 直接当线性权重,sum(|V|) 归一化避免 V 量级差异
- 类型 A (V=百万) 和类型 B (V=[-1,1]) 都被 sum-norm 自动处理
- 8 对独立处理 → 8 个 KV-aware tokens

## 序列编码:longer encoder + Top-K

baseline 已有 `LongerEncoder`:

1. 序列 [B, L, D] 输入
2. 提取最近的 K 步(`top_k=200`)
3. 对这 200 步做 self-attention
4. 输出 [B, K, D] 给 cross-attention 用

复杂度从 O(L²) 降到 O(K²)=40000,即使 L=1024 也能跑。

## 完整 EMA

```
each step:
  shadow[i] = 0.999 * shadow[i] + 0.001 * model[i]   for all parameters

each evaluation / checkpoint save:
  swap shadow → model      # eval / save EMA-version
  ... eval / save ...
  swap back               # restore for continuing training
```

shadow 包含**全部** 2.37 亿参数(emb + dense),约 1 GB fp32 显存开销。

## 关键超参

| 参数 | 值 | 来源 |
|---|---|---|
| d_model | 64 | baseline |
| emb_dim | 64 | baseline |
| num_queries | 2 | baseline |
| num_hyformer_blocks | 2 | baseline |
| num_heads | 4 | baseline |
| user_ns_tokens | 5 (KV-Aware: 5 个非 KV 部分,加 8 个 KV-aware = 13 user tokens) | W2 |
| item_ns_tokens | 2 | baseline |
| seq_top_k | 200 | **新**:longer encoder 压缩 |
| seq_max_lens | a:512,b:512,c:512,d:1024 | **新**:延到 p50 |
| dropout_rate | 0.01 | baseline |
| batch_size | 64 | **新**:EMA 显存平衡 |
| use_bf16 | True | W1 |
| use_ema | True (decay 0.999, **完整版**) | W1 |
| loss_type | bce | baseline |

## T 约束自动处理

T = num_queries × num_seqs + num_ns = 2×4 + 16 = 24,d_model=64,64%24≠0
→ 模型自动 fallback `rank_mixer_mode: full → ffn_only`(已实现,不需手动配)
