# millow_6 开发日志

## 决策依据

W2 sum-pool +0.0013, W3 attn pool -0.0095。**单一 pooling 信号弱**。

观察:V seq 只有 3-10 元素,每种 pool 各有得失。**让模型从 3 个视角看,而不是赌单一视角**。

## 实施步骤

1. 复制 millow_4 全套
2. `model.py` `KVAwareUserNSTokenizer.__init__`:
   - `kv_token_projs[i]` 输入维度从 `emb_dim` 改成 `3 * emb_dim`(其他不变)
3. `model.py` `KVAwareUserNSTokenizer.forward`:
   - 替换原 sum-pool 单视角为 3 view 计算
   - sum_view、max_view、mean_view 各 (B, emb_dim)
   - cat 成 (B, 3 * emb_dim) → kv_token_projs[i] → 1 token
4. 同步 `evaluation/model.py`
5. `train.py` / `evaluation/infer.py` **无需改动**(没有新 CLI flag)
6. `run.sh` 跟 millow_4 完全一样(关键改动在 model.py 内部)

## 预期 AUC 计算

```
millow_4(W2 sum 单视角)预期:0.820 中位
+ multi-view 增加表达力(经验值)
+0.005~0.010
= 目标 0.825 ~ 0.830
```

## 风险评估

| 风险 | 缓解 |
|---|---|
| max_view 用 argmax 不可微 | gather 可微(只对选中位置流梯度),验证可 backward |
| 全 padding 行 max_view 偏置(取 K_emb[0]) | 已用 `(~no_valid).float()` mask 强制清零 |
| 3 view 拼接训练初期 LayerNorm 数值不稳 | LayerNorm 已加在 kv_token_projs 内,无需额外处理 |
| 跟 evaluation 的兼容性 | 没新 ctor 参数,evaluation/model.py 同步即可,strict load 一致 |

## 部署清单

- [x] millow_6/train/model.py 改 KV-Aware multi-view
- [x] millow_6/evaluation/model.py 同步
- [x] millow_6/train/run.sh 配置(跟 millow_4 一样)
- [x] millow_6/evaluation/infer.py 不需改(已有 OOM 修复 from W3)
- [ ] (待提交) 平台训练 + 评估

## 提交策略

3 个版本如何分批提交(假设一天 3 个 Job):

| 时段 | 选 | 理由 |
|---|---|---|
| Day 1 上午 | **millow_4**(保底) | 撤回 W3,先确保有正向 |
| Day 1 下午 | **millow_5**(冲分) | DCN-V2 是被验证的 +1-2% 杠杆 |
| Day 1 晚 | **millow_6**(实验) | Multi-view pool 是新颖思路,赌中收益高 |

如果 millow_4 失败(显存爆 / 长序列效果差),Day 2 改超参重试 millow_4 而不是死磕 millow_5/6。
