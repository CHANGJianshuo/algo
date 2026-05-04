# millow_5 开发日志

## 关键决策

DCN-V2 是 CTR 任务被反复验证的"显式特征交叉"机制。millow_4 全靠 attention 隐式学,millow_5 把这部分**显式化**。

## 实施步骤

1. 复制 millow_4 全套(train + evaluation)
2. `model.py` 新增 `CrossNetV2` 类(40 行)
3. `PCVRHyFormer.__init__` 接收 `use_dcn / dcn_layers` 参数,在 `num_ns` 计算后构造 `self.dcn`
4. `PCVRHyFormer.forward / predict` 在 `ns_tokens = cat(ns_parts)` 后插入 DCN block
5. `train.py` 加 CLI flag + 透传到 `model_args`
6. `evaluation/infer.py` 的 `_FALLBACK_MODEL_CFG` 加 `use_dcn` 和 `dcn_layers`,确保 strict load 一致
7. `evaluation/model.py` 同步

## 预期 AUC 计算

```
millow_4 起点(预期):0.820
+ DCN-V2 显式二阶交叉:+0.005~0.010(经验值,Avazu/Criteo +1-2%,推 0.5-1.0% 保守)
= 目标 0.825 ~ 0.830
```

## 风险评估

| 风险源 | 缓解 |
|---|---|
| DCN-V2 1024 维向量 LayerNorm 数值不稳 | 加了 LayerNorm in cross,不会爆 |
| DCN 引入 ~2M 参数,小数据(900K)过拟合 | 训练时 dropout=0.01 有效;EMA 进一步稳化 |
| 跟 KV-Aware 信号冲突 | DCN 在 NS tokens 后做,不会破坏 KV-Aware 内部 pooling |
| 评估时 DCN 模块缺失 → strict load fail | 已在 _FALLBACK_MODEL_CFG 加 key,test 通过 |

## 关键代码行号

- `model.py` 加 `CrossNetV2` 类:line ~1459
- `PCVRHyFormer.__init__` use_dcn 接收:line ~1248
- `self.dcn` 构造:line ~1707
- forward DCN 应用(2 处):line ~2082(forward), line ~2137(predict)
- `train.py` flag:line ~228
- `evaluation/infer.py` _FALLBACK_MODEL_CFG:line ~76

## 部署清单

- [x] millow_5/train/model.py 加 CrossNetV2 + PCVRHyFormer 改
- [x] millow_5/train/train.py 加 CLI flag + model_args
- [x] millow_5/train/run.sh 启用 --use_dcn
- [x] millow_5/evaluation/model.py 同步
- [x] millow_5/evaluation/infer.py 加 _FALLBACK_MODEL_CFG keys
- [ ] (待提交) 平台训练 + 评估
