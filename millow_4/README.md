# millow_4 — 长序列 + W2 sum-pool + 完整 EMA

## 一句话目标

撤回 W3 attention pool(-0.0095 反向收益),回到稳定的 W2 sum-pool(+0.0013),叠加两个**未充分利用**的最大杠杆:**长序列** + **完整 EMA**。

## 历史结果回顾

| 版本 | 改动 | AUC |
|---|---|---|
| baseline | 主办方原版 | **0.8125** |
| W2 KV-Aware (sum-pool) | 8 对 KV 配对处理 | 0.8138 (+0.0013) |
| W3 KV attention pool | softmax 加权(失败) | **0.8043 (-0.0082)** |
| millow_4 | W2 sum-pool + 长序列 + 全 EMA | **目标 0.815~0.825** |

## 核心改动

### 1. KV pooling 回到 W2 sum-pool

`baseline/model.py KVAwareUserNSTokenizer.forward` 用 `weighted = sum(K_emb * V) / sum(|V|)`,删除 W3 的 attention head MLP。

**为什么撤 W3**:
- V 序列长度只有 3-10,**softmax over 3 个数字不稳定**
- attention MLP 引入新参数(~16K),5 个 epoch 早停内训不充分
- 实测 -0.0082(W2 0.8138 → W3 0.8043),不是噪声

### 2. 序列大幅延长

来自 demo 数据分析:

| 域 | 实际 p50 中位长度 | baseline 截断 | 信息损失 |
|---|---|---|---|
| domain_a | **578** | 256 | ~60% |
| domain_b | 405 | 256 | ~55% |
| domain_c | 322 | 512 | ~30% |
| domain_d | **1036** | 512 | ~50% |

millow_4 改:`--seq_max_lens seq_a:512,seq_b:512,seq_c:512,seq_d:1024`

→ d 域接近 p50,a/b/c 也至少 p25 以上。

为防止序列变长后 attention 显存爆炸,搭配:
`--seq_encoder_type longer --seq_top_k 200`

`longer` encoder(baseline 已实现)用 **Top-K 压缩**:每域 attention 只在最近的 200 步上计算,显存 ~O(L) 而非 O(L²)。

### 3. 完整 EMA(不是 dense_only)

`--use_ema --ema_decay 0.999`(默认完整版,包含 2.37 亿 embedding shadow)

- W1 dense_only EMA 失败的可能原因之一是只 EMA dense 部分(占总参数 1%),效果太弱
- 完整 EMA 对所有参数做 shadow,这才是 RecSys 文献里被验证有效的版本(典型 +0.3-0.8 AUC)
- batch_size 砍到 64 给 EMA shadow 留显存

## 文献依据

- **EMA**: 几乎所有 SOTA RecSys 论文(SASRec / BST / Wukong)都用 EMA 或 SWA 作为最终模型
- **长序列**: Meta GR(Generative Recommenders, 2024)证明序列长度从 256 → 8192 时 AUC 持续上升
- **撤回学习型 attn pool**: SimpleX (Mao et al., RecSys 2021)证明 simple pooling on short interaction sets often beats learned attention

## 文件夹结构

```
millow_4/
├── README.md          ← 本文档
├── ARCHITECTURE.md    ← 模型结构详解
├── DEVLOG.md          ← 开发记录
├── train/             ← 训练代码包(上传 Model Training)
│   ├── dataset.py     主办方原版
│   ├── ema.py         W1 EMA 类
│   ├── latency_harness.py  W1 后训练延迟测量
│   ├── model.py       **改动**: KVAware sum-pool 版
│   ├── ns_groups.json 主办方原版
│   ├── run.sh         **改动**: W3 配置
│   ├── train.py       W1/W2 CLI flag
│   ├── trainer.py     W1 EMA hooks
│   └── utils.py       主办方原版
└── evaluation/        ← 评估代码包(上传 Model Evaluation)
    ├── dataset.py     主办方原版
    ├── infer.py       eval batch=16 + bf16 修复
    └── model.py       与 train/model.py 同步
```

## 提交

1. 平台 Model Training:Job Name `millow_4`,上传 train/ 下 9 个文件,run.sh 已配好,直接 Submit
2. Resources = 1
3. 训完后 Model Evaluation:上传 evaluation/ 下 3 个文件,选 millow_4 的 model

## 风险

- **EMA 显存**:完整 EMA 多 ~1GB shadow,vGPU 可能紧张。已用 batch=64 + bf16 平衡
- **longer encoder Top-K**:K=200 可能不够(domain_d p50=1036 砍掉 80%),但相比 baseline 还是有进步
- **如果还 OOM**:回滚到 `--seq_max_lens seq_a:256,seq_b:256,seq_c:256,seq_d:512`,只保留 EMA
