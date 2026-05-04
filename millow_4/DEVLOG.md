# millow_4 开发日志

## 决策依据

W3 失败教训:在不充分理解之前,**不要把简单可行的方法替换为复杂的学习型版本**。
- W2 sum-pool 0.8138 是已验证有效的
- W3 改成 attention pool 引入新参数 + 不稳定的 softmax → -0.0095

millow_4 哲学:**保留有效部分(W2 sum-pool),只引入两个低风险高确定的改进**。

## 改动清单

| 改动 | 文件 | 风险 | 预期收益 |
|---|---|---|---|
| 撤回 W3 attn pool | model.py | 低(回到已验证版本) | +0.0095(撤回错误改动) |
| 序列延长 a/b/c/d → 512/512/512/1024 | run.sh | 中(显存压力) | +0.005~0.015 |
| `--seq_encoder_type longer --seq_top_k 200` | run.sh | 低 | 显存可控 |
| `--use_ema --ema_decay 0.999`(完整版) | run.sh | 中(显存增加 ~1GB) | +0.003~0.008 |
| `--batch_size 64` | run.sh | 略损 AUC | -0.001(平衡显存) |
| `--use_bf16` | run.sh | 极低 | 显存减半 |

## 预期 AUC 计算

```
0.8138 (W2 baseline 起点)
+ 0.010 (序列延长,中位估)
+ 0.005 (完整 EMA)
- 0.001 (batch 减半)
- 0.001 (bf16 精度)
= 0.8268 中位预期
```

合理区间:**0.815 ~ 0.825**

## 失败回滚预案

若 millow_4 跑出 < 0.8138(W2 不如):

| 现象 | 可能原因 | 回滚 |
|---|---|---|
| 显存不够,batch 都得砍到 16 | longer encoder 设置不对 | 改 `--seq_top_k 100` 或恢复 transformer encoder |
| AUC 大幅下降(<0.80) | 完整 EMA 不稳定 | 改 `--ema_dense_only`(只 EMA dense) |
| 训练发散 | 长序列 + bf16 数值不稳 | 去掉 bf16,batch=32 fp32 |

## 已知失败模式预防

- ❌ 不要再叠加 `--use_lr_warmup`(W1 试过,效果不明)
- ❌ 不要叠加 `--include_ns_in_classifier`(W1 试过,evaluation 还要修)
- ❌ 不要用 Focal Loss(正样本率 12% 不算极不平衡,BCE 即可)

## 代码同步检查

- [x] millow_4/train/model.py 改了 KVAwareUserNSTokenizer
- [x] millow_4/evaluation/model.py 同步
- [x] millow_4/train/run.sh 配置 EMA + 长序列
- [x] millow_4/evaluation/infer.py 已有 W3 OOM 修复(batch=16+bf16)

## 提交计划

| 时段 | 动作 |
|---|---|
| Day 1 | millow_4 提交训练(首选,稳健) |
| Day 1 同时段 | millow_5 提交训练(备选,DCN-V2 探索) |
| Day 2 | 看结果决定 millow_6 是否提交,或基于 millow_4/5 winner 微调 |
