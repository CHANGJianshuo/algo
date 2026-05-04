# millow_5 — DCN-V2 显式特征交叉 + W2 KV + 长序列 + EMA

## 一句话目标

在 millow_4 基础上**加 DCN-V2 cross network**(Wang et al., WWW 2021)显式建模 NS tokens 之间的特征交叉。CTR 任务里这是经典的 +0.5%~2% AUC 杠杆。

## 历史与定位

| 版本 | 改动 | AUC | 关系 |
|---|---|---|---|
| baseline | 主办方原版 | 0.8125 | 起点 |
| W2 sum-pool | KV-Aware 8 对 | 0.8138 | millow_5 KV 实现的依据 |
| W3 attn pool | KV softmax | 0.8043 | 撤回 |
| millow_4 | W2 + 长序列 + EMA | 目标 0.815~0.825 | millow_5 同基础 |
| **millow_5** | **+ DCN-V2 cross** | **目标 0.820~0.830** | 加显式交叉 |

## 核心新增

### DCN-V2 Cross Network

每层做:`x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l`

- **x_0**:flatten 后的 NS tokens 向量 `(B, num_ns × d_model) = (B, 16 × 64) = (B, 1024)`
- **x_l**:第 l 层的输出
- **逐元素乘** (`⊙`) 实现显式高阶特征交叉
- **残差连接** + LayerNorm 保留原信号 + 防数值发散

baseline 里 NS tokens 之间的交互完全靠 HyFormer attention 学习,**实现"二阶交叉" 隐式而低效**。DCN-V2 把"二阶 / 三阶交叉"做成**显式数学操作**,模型直接拿到这个表征,attention 可以专注于序列侧的复杂模式。

### 实现位置

`PCVRHyFormer.forward` 里,**NS tokens cat 完之后、HyFormer 之前**加一段:

```python
ns_tokens = torch.cat(ns_parts, dim=1)         # (B, num_ns, D)
if self.dcn is not None:
    flat = ns_tokens.reshape(B, -1)             # (B, num_ns * D)
    crossed = self.dcn(flat)                    # (B, num_ns * D)  DCN-V2
    ns_tokens = crossed.reshape(B, num_ns, D)   # 回到 (B, num_ns, D)
```

DCN 层数:2(默认)。1024 维 × 2 层 ≈ 2M dense 参数,相比 2.37 亿 emb 参数微不足道。

## 文献依据

- **DCN-V2**: Wang et al., "DCN V2: Improved Deep & Cross Network ...", WWW 2021. Google production CTR 模型核心组件,在 Avazu、Criteo benchmarks 上稳定 +1-2%。
- **特征交叉对 CTR/CVR 的重要性**: 工业 RecSys 共识 — 用户性别 × 商品类目 这种二阶交叉是核心信号,用 attention 隐式学很慢。
- **DCN-V2 vs FM/DeepFM**: V2 用 vector × matrix 替代 V1 的 vector × vector,表达力更强,且 LayerNorm 让训练更稳。

## 关键超参

跟 millow_4 相同 +:
- `--use_dcn` ON
- `--dcn_layers 2`

## 风险与回滚

| 风险 | 表现 | 回滚 |
|---|---|---|
| DCN 训练发散 | loss 早期 NaN | 减层数 `--dcn_layers 1` |
| 过拟合 | val AUC < train AUC 差距大 | 加 `--dropout_rate 0.05` |
| 显存爆 | OOM | `--seq_top_k 100` 或 `--batch_size 32` |
| DCN 没用 | 跟 millow_4 持平甚至降 | 关闭 `--use_dcn`,该方向放弃 |

## 提交

Job Name `millow_5`。上传 train/ 8 个文件 + evaluation/ 3 个文件。`run.sh` 已配好。
