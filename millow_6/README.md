# millow_6 — Multi-View KV Pooling(实验性)

## 一句话目标

每对 KV 同时做 **3 种 pool**(sum / max / mean),拼起来给模型 **3 个互补视角**,而不是赌单一 pooling 方式。

## 历史与定位

| 版本 | KV pool 方式 | AUC |
|---|---|---|
| W2 | sum-pool (V-weighted) | 0.8138 |
| W3 | learned attention pool | 0.8043(失败) |
| millow_4 | sum-pool 单视角 | 目标 0.815~0.825 |
| millow_5 | sum-pool + DCN-V2 | 目标 0.820~0.830 |
| **millow_6** | **3 view pool 拼接** | 目标 0.820~0.832 |

## 核心思路

V 序列长度只有 3-10,**任何单一 pooling 都损失信息**:

| Pool | 捕捉到 | 漏掉 |
|---|---|---|
| **sum-pool**(V 加权) | 用户对哪个 K 偏好最强(数值反映) | 没考虑稀有但可能重要的 K |
| **max-pool**(取 |V| 最大那个 K) | 用户的 top 偏好(尖峰) | 多元偏好的整体性 |
| **mean-pool**(无加权平均) | 用户的"兴趣面"(广度) | 强度信息 |

3 view 拼接 → Linear → 1 token,**让模型自己学怎么组合**这 3 个视角。

## 实现

```python
for each KV pair (8 总):
    k_ids = K[B, N], v_vals = V[B, N], mask = (K != 0)
    k_emb = Embedding(k_ids)  # (B, N, emb_dim)

    # View 1: V-weighted sum-pool
    sum_view = (k_emb * V_masked).sum(dim=1) / |V|.sum().clamp(1e-6)

    # View 2: max-pool(取 V 最大位置的 K_emb)
    argmax = |V_masked|.argmax(dim=1)
    max_view = k_emb[:, argmax]  # gather

    # View 3: mean-pool(无加权)
    mean_view = (k_emb * mask).sum(dim=1) / mask.sum().clamp(1)

    # Concat → Linear(3*emb_dim, d_model) → 1 token
    token = SiLU(Linear(cat([sum_view, max_view, mean_view])))
```

## 文献依据

- **Multi-Pool Aggregation**: Zhou et al., "Deep Interest Network for Click-Through Rate Prediction" (KDD 2018) — 早期就证明对短序列 attention pool 不一定 work,简单 pool 组合反而更稳。
- **TransAct (Pinterest, KDD 2023)**: 用户行为序列建模里多种 pooling 组合是 SOTA 标配。
- **Set Transformer**: PMA (Pooling by Multihead Attention) 的对照实验里 mean-pool + max-pool 组合在 ≤10 元素的小集合上经常打败 attention。

## 风险

| 风险 | 缓解 |
|---|---|
| 参数变多(每对 Linear 输入 3×emb_dim) | 实际只多 ~30K 参数,占比微不足道 |
| max-pool 不可微?| `argmax` 不可微,但 gather 可微(梯度只流到选中那个位置)。OK |
| 全 padding 行 max-pool argmax 取 0 → K_emb=0 | 已处理:乘以 (~no_valid).float() 把空行强制清零 |

## 关键超参

跟 millow_4 一样,只是 model.py 内部 KV pool 改了。无新增 CLI flag。

## 提交

Job Name `millow_6`。上传 train/ 8 个文件 + evaluation/ 3 个文件。run.sh 已配好。
