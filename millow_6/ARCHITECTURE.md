# millow_6 架构

## KV-Aware 内部数据流(Multi-View)

```
For each KV pair (fid 62-66, 89-91; 共 8 对):

  K_ids   ∈ ℤ^(B × N)        (整数 ID,长度 3-10)
  V_vals  ∈ ℝ^(B × N)        (浮点权重,跟 K 对齐)
       │
       ▼
  k_emb = Embedding(K_ids)   ∈ ℝ^(B × N × emb_dim)
  mask  = (K_ids != 0).float()
  v_msk = V_vals * mask
       │
       ├─────────┬─────────┬─────────┐
       ▼         ▼         ▼         │
   ┌────────┐ ┌────────┐ ┌────────┐  │
   │sum-pool│ │max-pool│ │mean-pool│ │
   │(V 加权)│ │(top|V|) │ │(无权均) │ │
   └───┬────┘ └───┬────┘ └───┬────┘  │
       │          │          │       │
       └──────────┼──────────┘
                  ▼
              concat
                  │
                  ▼
          (B, 3*emb_dim) = (B, 192)
                  │
                  ▼
       Linear(192, d_model=64) + LayerNorm + SiLU
                  │
                  ▼
              1 KV-aware token (B, 1, 64)

× 8 对 = 8 KV-aware tokens
```

## 跟 millow_4(单视角)的差异

| 维度 | millow_4 | millow_6 |
|---|---|---|
| KV pool 数量 | 1(sum 加权) | 3(sum + max + mean) |
| 输入维 → 输出维 | emb_dim → d_model | 3 × emb_dim → d_model |
| 每对参数 | ~4K | ~12K |
| 每对计算 | 1 次 sum + 1 次 norm | 3 种 pool 同时算 |

## 整体网络

```
NS Tokenizer:
  - User: 8 KV-aware (multi-view) + 5 non-KV RankMixer = 13 user tokens
  - User dense (non-KV: fid 61, 87) = 1 token
  - Item: 2 RankMixer tokens
  Total NS: 16 tokens (跟 millow_4/5 一致)

Sequence Encoders (longer + Top-K 200):
  4 domains, seq_max_lens = a:512, b:512, c:512, d:1024
  
Query Generator: 8 query tokens(每域 2)

HyFormer Block × 2 (rank_mixer ffn_only fallback)

→ 8 query tokens → output_proj → 1 logit → BCE
```

## 内存对比

每对 KV 多算 2 个 view(max, mean),增量 forward 计算可忽略。

参数增量:`8 pair × 2 × emb_dim × d_model ≈ 8K`,相对 2.4 亿总参数微不足道。

## 跟 millow_5 的对比

millow_5 在 NS tokens 之间加 DCN-V2 显式交叉(横向交叉);
millow_6 在每对 KV 内部用多视角(纵向丰富化)。

理论上**两者可以叠加**(下一步 millow_7 可考虑),但当前每个版本独立测试以观察净贡献。
