# TAAC2026 学术赛道 — 决赛 Top 30 + 创新奖执行方案

## Context

**确定参数**:
- 赛题:Towards Unifying Sequence Modeling and Feature Interaction (单 block 输出 pCVR,AUC 唯一指标)
- 赛道:**学术赛道,1M 样本**
- 算力:**多卡 A100/H100,≥8 卡**
- 目标:**决赛 Top 30 + Unified Block / Scaling Law 创新奖($45K × 2)**
- 起点:**完全从零开始**

**三条硬约束**(架构与代码必须围绕这三条):
1. 严禁 ensemble — 单模型决胜负,EMA 是唯一合法"伪平均"
2. 推理延迟硬上限 — 超时直接 DQ
3. 1M 样本上限 — 模型容量与正则需精细平衡,过参数会过拟合

**数据规格**(120 列扁平):
- 5 ID/Label: `user_id, item_id, label_type, label_time, timestamp`
- 46 User Int(35 标量 + 11 数组)
- 10 User Dense:
  - `user_dense_feats_61` = SUM 用户表征
  - `user_dense_feats_87` = LMF4Ads(Tencent 大基座,arxiv 2508.14948)
  - `user_dense_feats_{62-66, 89-91}` = 与对应 `user_int_feats` 逐元素对齐(KV 对结构)
- 14 Item Int(13 标量 + 1 数组 `item_int_feats_11`)
- 45 Domain Sequences(A=9, B=14, C=12, D=10)

---

## 一、最终架构选型(锁定)

### 主干:**HSTU-Unified Block + KV-Aware Token Fusion**

**为什么是它**:
- HSTU(Meta GR)已被证明在序列推荐 scaling 良好,天然契合赛题"统一 block + scaling law"主题
- 1M 数据 + 8 卡 → 50-150M 参数是甜点区,HSTU 6-12 层正合适
- Gated Linear Unit 同时具备序列建模与特征交叉能力,真正"同构"
- 线性复杂度对延迟友好,留出量化/编译优化空间

### Token 化方案(关键创新点)

将 120 列异构特征统一编码为单一 token 序列:

```
[CLS] [USER_SCALAR_TOKENS×35] [USER_KV_TOKENS×8(对齐字段聚合)]
      [USER_ARRAY_TOKENS×N(变长)] [USER_DENSE_BACKBONE(SUM+LMF4Ads, adapter 注入)]
      [ITEM_SCALAR_TOKENS×13] [ITEM_LABEL_TOKENS(item_int_feats_11)]
      [DOMAIN_A_SEQ×L_a] [DOMAIN_B_SEQ×L_b] [DOMAIN_C_SEQ×L_c] [DOMAIN_D_SEQ×L_d]
```

每个 token 由三类 embedding 相加:
- **内容 embedding**:特征值的查表 / 投影
- **字段 type embedding**:120 维 type vocab,告知 token 来自哪个字段
- **位置/时间 embedding**:序列 token 用相对时间间隔(t_now - t_event)分桶

### KV-Aware Fusion(`user_int_feats_{62-66,89-91}` 与对齐 dense)

不可展平丢失结构。处理方式:
```
key_emb = embed(int_feat_value)        # [N, d]
val_proj = linear(dense_feat_value)    # [N, d]
kv_token = key_emb + val_proj          # 或 gated: key * sigmoid(val_proj)
field_token = pool(kv_tokens)          # attention pool over N
```

每个对齐字段产出 1 个 token。这是相对其他队伍的关键差异化点。

### Domain Sequence 处理

- 4 域共享 item embedding 表(避免 1M 样本下 4× vocab 爆炸)
- 每个域加 domain_id embedding 区分
- 序列长度统一截断到 **128**(scaling law 实验时 sweep:32/64/128/256)
- **目标 item 与每个域序列做 cross-attention**,产出 4 个 domain-specific 表征
- 4 个表征 + 全局 CLS 共同参与最终预测

### 创新模块:**Domain-Routed Gated HSTU Block**(冲创新奖核心)

在 HSTU block 内插入一个**域路由门控**:
```python
# 每个 token 根据其域/类型 id 学习一个混合权重
g = softmax(W_route · type_emb)        # [num_paths]
out = g[0]·sequence_path(x) + g[1]·feature_interaction_path(x) + g[2]·identity(x)
```
- `sequence_path`:HSTU GLU(沿序列维)
- `feature_interaction_path`:DCN-V2 风格的 bit-wise crossing(沿字段维)
- `identity`:残差直通

这样**同一个 block** 既能消化序列又能消化特征交叉,通过门控自动路由 — 直接对应 "Unified Block" 主题,论文卖点鲜明。

---

## 二、训练范式(锁定)

### 损失
- 主损失:BCE on `label_type`
- 辅助损失(消融后保留涨点项):
  - **LMF4Ads 蒸馏**:模型 user 隐层向 `user_dense_feats_87` cosine 对齐(权重 0.1)
  - **跨域对比**:同 user 不同域序列表征 InfoNCE 拉近(权重 0.05)
  - **Item multi-label 预测**:`item_int_feats_11` 作为 multi-hot 辅助任务(权重 0.05)

### 数据划分
- 严格按 `timestamp` 时间切分,**最后 15% 作 valid**
- 禁止 random KFold(时序泄漏)
- 留 **最后 5% 作 holdout**,只在最终选模型时看一次

### 防泄漏 checklist(每周 review)
- [ ] 序列字段中所有 event 的 timestamp < 样本 label_time
- [ ] 任何 target encoding/统计特征只用 valid 切分前的窗口计算
- [ ] 同一 (user_id, item_id) 不在 train/valid 同时高频出现

### 优化器
- AdamW,base lr 1e-3,embedding lr 3e-3
- Cosine LR + 5% warmup
- bs=4096(按 seq_len 调),梯度累积到 16384
- **EMA decay 0.999**(必备)
- Stochastic Depth 0.1,DropPath 0.1
- Label smoothing 0.05
- BF16 训练 + FP32 主权重

### Scaling Law 实验矩阵(冲 Scaling Law 创新奖)

固定其他变量,各跑 4-5 个点:

| 维度 | 取值 | 实验数 |
|---|---|---|
| Params (depth × dim) | 10M / 30M / 60M / 120M / 200M | 5 |
| Seq length | 32 / 64 / 128 / 256 | 4 |
| Embed dim | 64 / 128 / 192 / 256 | 4 |
| Train tokens (epoch) | 0.5× / 1× / 2× / 4× data | 4 |

每点跑 3 个 seed 取均值,共 ~50 次训练,8 卡 A100 上 2 周可完成。最终在技术报告里画 4 张对数线性图。

---

## 三、延迟工程(决胜约束)

### 第 1 周必交付:Latency Harness
脚本 `tools/latency_probe.py`,做到:
- 模拟官方评测环境的 batch size 与硬件
- 每次架构改动前后跑一次,记录 p50/p99 延迟
- 与 budget 对比,超阈值红灯

### 提速优先级(开发顺序)
1. BF16 推理(默认开)
2. FlashAttention(HSTU 自带 GLU 兼容)
3. 序列长度 ≤128
4. torch.compile(model, mode='reduce-overhead')
5. INT8 PTQ(最后 1 周再上,有精度风险)
6. 自蒸馏到小模型(若大模型超时,只能走这条)

### 红线
- 模型层数 ≤ 12
- Seq length ≤ 256(超长走 SIM 两阶段检索)
- 最终推理参数量 50-150M

---

## 四、代码骨架(从零起步)

```
taac2026/
├── configs/
│   ├── base.yaml               # 默认超参
│   ├── ablation_*.yaml         # 各消融实验
│   └── scaling_law_*.yaml      # scaling sweep 配置
├── data/
│   ├── loader.py               # parquet 流式加载,arrow 内存映射
│   ├── tokenizer.py            # 120 列 → token 序列
│   ├── stats.py                # vocab 统计、缺失值/长尾分析
│   └── splits.py               # 时间切分 + holdout
├── models/
│   ├── embeddings.py           # 共享 vocab,KV-fusion,LMF4Ads adapter
│   ├── hstu_block.py           # HSTU GLU 实现
│   ├── routed_block.py         # 创新点:Domain-Routed Gated HSTU
│   ├── unified_model.py        # 整体 forward
│   └── losses.py               # BCE + 蒸馏 + 对比 + multi-label
├── train/
│   ├── trainer.py              # 主训练循环 + EMA + AMP
│   ├── eval.py                 # AUC + 延迟测量
│   └── ddp.py                  # 8 卡 DDP 启动
├── tools/
│   ├── latency_probe.py        # 延迟 harness
│   ├── leak_detector.py        # 时序泄漏检查
│   ├── infer_submit.py         # 生成提交文件
│   └── scaling_plot.py         # 出 4 张 scaling law 图
├── experiments/
│   └── runs/                   # mlflow / wandb 日志
└── report/
    ├── tech_report.tex         # 技术报告(创新奖必交)
    └── ablation_table.md       # 维护至少 8 行的 ablation 表
```

### 复用的已有实现(避免重造)
- HSTU 参考实现:Meta `generative-recommenders` repo
- DCN-V2:DeepCTR / RecBole 中已有
- FlashAttention:`flash-attn` 包
- EMA / DropPath:`timm.utils`
- Parquet 大数据加载:`pyarrow.dataset` 流式

---

## 五、9 周排期

| 周 | 关键任务 | 验收标准 |
|---|---|---|
| **W1** | 环境 + 数据加载 + EDA + Latency Harness + Baseline DNN(DIN-style) | 提交跑通,LB 有分;延迟脚本可用;EDA 报告完成 |
| **W2** | KV-Aware Fusion + LMF4Ads/SUM Adapter,baseline 强化 | AUC 较 W1 提升 ≥ 1.5pt |
| **W3** | HSTU 主干替换 MLP 交互 | 较 W2 提升 ≥ 1pt,延迟仍合规 |
| **W4** | Domain-Routed Gated HSTU(创新点)+ 跨域 cross-attn | 较 W3 提升 ≥ 0.5pt;消融表 ≥ 4 行 |
| **W5** | 辅助损失消融(蒸馏/对比/multi-label) | 确定 1-2 个保留,消融表 ≥ 6 行 |
| **W6** | Scaling Law 系统实验(50 次训练) | 4 张对数线性图就绪 |
| **W7** | 超参精调 + EMA + LR/dropout sweep | LB 进 Top 30 |
| **W8** | 延迟优化(compile/FA/量化)+ 蒸馏备选 | 延迟剩余 ≥ 30% 余量 |
| **W9** | 最终复现 + 技术报告写作 + 提交 | 报告完成,代码可一键复现 |

---

## 六、防过拟合(1M 样本核心风险)

- 模型 ≤ 150M 参数;Scaling Law 实验显示 60-100M 是 1M 数据下的甜点
- Embedding dropout 0.2,层 dropout 0.3
- Label smoothing 0.05,Mixup 仅在 embedding 层 α=0.1
- 每隔 200 步在 valid 评估,patience=5 早停
- 长尾 ID(出现 < 3 次)统一映射到 OOV bucket
- 不对 train+valid 一起做 vocab 统计 — 只用 train 统计

---

## 七、防泄漏 / 一致性 sanity 测试

每个 milestone 必跑:
- [ ] `tools/leak_detector.py`:序列 timestamp < label_time
- [ ] 本地 valid AUC vs LB AUC 散点图,r > 0.95
- [ ] holdout 与 valid 差距 < 0.005,否则 valid 已过拟合
- [ ] 关掉序列特征 → AUC 应大幅下跌(>2pt),证明序列在起作用
- [ ] 只用 LMF4Ads + LR → 看 baseline 上限,作 sanity

---

## 八、创新奖材料维护(从 W1 起每周更新)

- `report/ablation_table.md`:每个组件 on/off 的 AUC,目标 ≥ 8 行
- `report/scaling_logs/`:每次 scaling 实验的 loss / AUC 曲线
- `report/design_doc.md`:Routed HSTU 的动机、与 DIN/DCN 双轨架构的对比
- 配图脚本 `tools/scaling_plot.py` 输出 PDF 可直接插入论文

---

## 九、关键风险与对策

| 风险 | 对策 |
|---|---|
| 误用 ensemble | 全程单模型,EMA 是唯一合法"平均",CI 检查 commit 中无多模型加权代码 |
| 延迟超限被 DQ | W1 即建 latency harness,每次 PR 必跑;ckpt 同时保存大/小两份模型 |
| 序列泄漏 | leak_detector 进入 CI,每次数据 pipeline 改动必跑 |
| 1M 数据下大模型过拟合 | Scaling Law 实验找甜点;若 200M 反而下降则报告中明确写出 |
| LMF4Ads 用法不当 | W2 sanity:仅用 LMF4Ads + LR 看上限;adapter 与冻结策略 ablation |
| 跨域负迁移 | Domain-routed gating 配置中可关闭跨域 attn;消融对比 |
| 创新奖材料不足 | 从 W1 起强制每周更新 ablation_table.md |

---

## 十、立刻执行的下一步(W1 day 1-3)

1. 拉数据集到本地存储,确认 1M parquet 完整性
2. 搭 `taac2026/` 骨架,初始化 git + DDP + wandb
3. 写 `data/loader.py` + `data/stats.py`,产出 EDA 报告(各字段分布、缺失率、序列长度分布、标签正负比)
4. 写 `tools/latency_probe.py` 占位(真实环境暂不可用就 mock)
5. 写最简 baseline:embedding + mean pool + 3 层 MLP,提交一次拿初始 LB

## 验证方式

最终方案落地后需通过这些检查:
- 单模型 AUC 在 LB 进 Top 30
- 推理延迟 ≤ budget × 70%
- 4 张 Scaling Law 图全部对数线性
- Ablation 表 ≥ 8 行,Routed HSTU 各路径门控权重可视化
- 技术报告 ≤ 8 页,包含动机/架构图/消融/scaling/局限性

## 关键参考(技术报告必引)

- HSTU: Meta GR, "Actions Speak Louder than Words"(SIGKDD'24)
- LMF4Ads: arxiv 2508.14948(Tencent)
- TAAC2025 报告: arxiv 2604.04976
- DCN-V2: "Improved Deep & Cross Network", WWW'21
- DIN/SIM: 阿里目标注意力序列建模经典
- "Ads Recommendation in a Collapsed and Entangled World": arxiv 2403.00793
- FlashAttention-2: Tri Dao

---

# 十一、精调复审 — 潜在 bug、加固点与差异化收紧

针对 5 个核心问题做一次"红队"式复审。**这是对前面方案的修正层,实施时若有冲突以本节为准。**

## 11.1 HSTU + KV-Aware Token Fusion 的潜在 bug

### Bug 1:KV token 数爆炸
- 8 个对齐字段(`user_int_feats_{62-66, 89-91}`),每条样本中数组长度可达 N=数十到数百
- 若每个 KV 对独立成 token,8 字段 × 平均 50 = **400 token**,叠加 4×128 序列 + 50 标量 = 总 token >1000
- **后果**:O(L²) attention 显存×4,延迟×4,直接超 budget

**修正**:KV 字段不展平成多 token,而用**字段内 attention pool**:
```
keys = embed(int_arr)            # [N, d]
vals = linear(dense_arr)         # [N, d]
gates = sigmoid(W_g · vals)      # [N, 1]
field_token = sum(keys * gates) / max(sum(gates), 1e-6)   # [d]
```
每字段产出 1 个 token,8 字段 → 8 token。这同时是创新点(不是简单 mean pool,是 value-driven gated pool)。

### Bug 2:数组型 user int feats(`{15, 60, 62-66, 80, 89-91}`)长度不齐
- 11 个 array 字段,每条样本长度不同
- 若用 padding 到 max_len 会浪费,用 ragged tensor 又难配 FlashAttention

**修正**:
- 字段 62-66, 89-91 走 11.1 的 attention pool(已解决)
- 字段 15, 60, 80 单独做 multi-hot embedding pooling(mean + max + attention 三路 concat)
- 长度 P95 截断,P95 通常远小于 max,显存压力小

### Bug 3:type embedding 与位置 embedding 冲突
- 序列 token 需要相对时间编码(`Δt = label_time - event_time` 分桶)
- 非序列 token 没有时间概念,直接给 `Δt=0` 会让 attention 误以为它们都是"刚刚"
- **修正**:为非序列 token 用单独的 type-positional encoding,与序列 token 的时间编码使用**正交子空间**(各占 d/2 维)

### Bug 4:FlashAttention 与异构 mask 不兼容
- HSTU 主要在序列 token 间走 attention
- 非序列 token 是否参与 self-attention?若全部参与 → mask 矩阵复杂,FA 加速有限
- **修正**:采用**两段式架构**
  1. 第一段(N₁ 层):非序列 token + 4 域序列各自做局部 attention(可并行,FA 友好)
  2. 第二段(N₂ 层):全 token 做 fused attention(此处 token 数已被 11.1 压缩到 ≤300,可走 FA)

### Bug 5:embedding vocab 共享 vs 独立
- 4 域序列若共享 item embedding,假设了"item 在不同域行为语义一致"
- 若不同域是 e.g. {电商点击, 视频观看, 广告展示, 搜索} —— 同一 item_id 在不同域的语义可能不同
- **修正**:**共享主表 + 域 adapter**:`emb_d = emb_shared + W_d · emb_shared`,W_d 是低秩矩阵(rank=8),既共享又区分

## 11.2 Domain-Routed Gated HSTU 的创新性是否够拿 Unified Block Award

### 诚实评估
| 已有工作 | 与本方案重叠点 |
|---|---|
| HSTU 本身 | 已声称统一序列与特征,GLU 同时做两件事 |
| MoE / Switch Transformer | 路由门控不算新 |
| PEPNet (KuaiShou) | 个性化门控网络,已用于 CTR |
| HiNet (KuaiShou) | 多场景门控 |
| FuxiCTR | 多模块组合的统一框架 |

**结论**:仅仅"按 token 类型路由 sequence-path/feature-path"创新性中等,容易被审稿质疑成 PEPNet + HSTU 拼接。

### 加强差异化(三选一,推荐 ①②叠加)

**① Recency-Aware Routing(基于时间的动态路由)**
- 路由 gate 不止依赖 token type,还依赖 token 的"新鲜度"
- 直觉:近期行为走 sequence path(强调时序),远期/聚合特征走 feature-cross path(强调统计交叉)
- 公式:`gate = softmax(W·[type_emb, log(1+Δt)_bucket_emb, content_emb])`

**② Cross-Path Bilinear Mixing(路径间显式二阶交互)**
- 不仅仅加权和,引入路径间的 bilinear:`out = α·seq + β·cross + γ·(seq ⊙ W_mix · cross)`
- 第三项让两条路径的输出**显式相乘**,符合"特征交叉"的几何直觉
- 这一项是真正的原创点,暂未在公开文献找到完全相同形式

**③ Routing Sparsity Regularizer**
- 加 L0/Gini 正则鼓励 gate 在不同 token 上呈现差异化分布
- 防止 gate 退化成"所有 token 都走同一路径"
- 这一项让你在 ablation 表上能展示"gate 学到了什么"——可视化超 sell

### 关于 Scaling Law 创新奖
原计划 4 维 sweep(params, seq_len, embed_dim, train_tokens)是**够的最低配**。建议补一维:
- **路由稀疏度 vs 模型大小**:在不同模型规模下,gate 倾向是否变化?这是把 routing 创新与 scaling 绑定的关键实验,直接服务两个奖项。

## 11.3 9 周排期的现实性

### 风险点逐周
| 周 | 风险 | 调整 |
|---|---|---|
| **W1** | "环境+数据+EDA+Latency Harness+Baseline+首次提交"6 件事 7 天太满 | 拆:**W0**(数据下载+环境+repo 骨架,1 天) + W1 仅做 EDA+Latency Harness+最简 baseline,**首次提交推到 W2 周一** |
| W2 | KV-Aware Fusion 涉及自定义算子,debug 耗时 | 预留 1 天 debug buffer,削减目标到 ≥1.0pt 提升(原 1.5pt) |
| W3 | HSTU 主干替换,与 W2 改动叠加 | 必须在 W2 末有清晰 commit hash 作 baseline,W3 只改主干一处 |
| W4 | 创新模块涉及训练稳定性问题 | 先验证 ① Recency Routing 单独效果,**②③ 留到 W5** |
| W5 | 辅助损失消融 4 项 × 至少 2 配置 = 8 次训练 | 8 卡可并行 8 次,1 天内完成,但需在 W4 末固化训练流水 |
| **W6** | Scaling Law 50 次训练 | **8 卡并行 → 50/8 ≈ 7 批 × 4-6h = 28-48h**,2 天足够;但需独立测试集,不能动 valid |
| W7 | 调参容易过拟合 valid | 用 holdout(留出的 5%)只看一次,不参与超参选择 |
| W8 | 量化精度损失可能 ≥0.005 AUC | 提前在 W6 验证 INT8 精度;若不达标,提前训蒸馏 student |
| **W9** | 报告压缩到最后一周 | **报告写作并行启动于 W6**,W9 只做润色与图表 |

### 关键漏掉的 milestone
1. **LB 提交节奏管理**:多数比赛每日提交受限(典型 5 次/天)。需为每周预留至少 2 次"诊断性"提交(对照实验需配对提交)
2. **数据回归测试**:每次 data pipeline 改动后必须跑回归 — 否则 W4 之后的提升可能源于数据 bug
3. **复现性 freeze 点**:W7 末必须有可一键复现的 commit;W8 只允许 inference-side 改动
4. **官方推理协议适配**:估计 1-3 天工作量,需提前确认是 PyTorch/ONNX/TVM 何种格式,**放进 W6**

## 11.4 防泄漏 / 防过拟合 加固

### 已覆盖 → 已加固
| 项 | 加固动作 |
|---|---|
| 时间切分 | 增加:确认 `timestamp` 与 `label_time` 单位(看示例值约 1.77e9 → 秒级 Unix);若不一致需统一 |
| 序列泄漏 | 增加:对每条样本检查 max(seq_timestamps) < label_time,违反样本直接抛错 |
| Vocab 泄漏 | 增加:vocab 只用 train 统计;valid/test 中未见 ID 全部 → OOV bucket |

### 新增必做
1. **预训练 embedding 时间穿越检查**
   - `user_dense_feats_61` (SUM) 和 `user_dense_feats_87` (LMF4Ads) 是预训练得到的
   - 若 SUM/LMF4Ads 训练数据时间窗 ≥ 你的 valid 时间,等于变相泄漏
   - **动作**:在 W2 写一份 ablation,关掉这两个 dense feature 后 AUC 下降幅度若 >5pt,就要警惕
2. **Adversarial Validation(分布漂移诊断)**
   - 训一个二分类:train→0, valid→1
   - 若 AUC > 0.6,说明 train/valid 分布差异大,需用其结果做 sample weighting 让 train 接近 valid
3. **Cold User / Cold Item 不剔除**
   - valid 中只在 valid 出现的 user/item 是真实测试场景,保留并报告冷启动 AUC 单独数值
4. **Embedding-level Mixup 风险**
   - 计划提到 α=0.1 的 embedding Mixup;但 ID 类 embedding 做 Mixup 在语义上是错的(ID 之间没线性空间)
   - **修正**:Mixup 仅作用于 **dense feature** 与 **池化后的序列表征**,不作用于 ID embedding
5. **正负比与 Focal Loss**
   - CVR 任务正样本通常 <5%;若用纯 BCE,模型偏向预测多数类
   - **加备选**:`focal_loss(γ=1.5)` 与 `class_balanced_BCE`,在 W5 与 BCE 做 A/B
6. **EMA 选择策略**
   - 不要只用 EMA 模型评估,**同时记录原始权重 valid AUC**,选两者中较高者作为最终模型(保留快照,不算 ensemble)

### 减少过拟合
- 1M 样本 / 100M 参数 ≈ 10:1 token-to-param ratio,远低于 Chinchilla 最优 20:1
- **结论**:不要无脑追求 200M 参数,Scaling Law 实验大概率会显示 60-100M 是甜点
- 若希望大模型涨点,只能靠**预训练**(用 LMF4Ads 蒸馏 + 自监督序列预训练),而非纯监督扩参

## 11.5 与 SOTA 的差异化定位收紧

### 诚实对照表
| 工作 | 创新点 | 本方案差异 |
|---|---|---|
| **HSTU** (Meta) | GLU 替代 softmax,序列推荐 scaling | 我们**借用** HSTU 作主干,不能声称是创新 |
| **TIGER** | 语义 ID + 生成式预测下一个 item | 完全不同范式,我们是判别式 CTR/CVR 头 |
| **DCN-V2** | bit-wise cross,纯特征交互 | 我们集成此思路到 feature-cross path,引用而非创新 |
| **PEPNet/HiNet** (Kuai) | 个性化/多场景 gating | 我们 routing 维度更细(per-token 而非 per-sample) |
| **FuxiCTR** | CTR 模块化框架 | 我们是 unified block,非框架 |
| **PinnerFormer / SASRec** | 序列推荐主干 | 不处理多字段特征交叉 |

### 真正的差异点(写论文时强调这三条)

1. **Per-Token Routing within a Single Block**
   - PEPNet/HiNet 的 gating 在 sample 或 task 级
   - 本方案在 token 级,且是同一 block 内 sequence-path / feature-path 的路由
   - 这是 MoE 的 token-level 形式,但 expert 不是 FFN 而是**两条结构性不同的算子路径**

2. **KV-Aware Field Tokenization**
   - 现有工作几乎都把 dense feature 当独立标量 concat
   - 本方案显式建模 `(int_id, dense_value)` 的 key-value 对齐结构,用 value-driven gated pool
   - 这一点在 ablation 中应能呈现 ≥1pt AUC,**写论文时单列一节**

3. **Routing Sparsity ↔ Scaling Law**
   - 大多数 unified 工作不研究 routing 决策如何随规模演变
   - 本方案在 4 维 sweep 之外加 routing entropy 维度,呈现**模型变大时门控如何变稀疏/专门化**
   - 这把 Unified Block 与 Scaling Law 两个奖项绑定,论文卖点统一

### 不应声称的差异化(避免审稿被打)
- ❌ "首个统一 block":HSTU 已在做
- ❌ "首次跨域统一":CDR 领域工作多
- ❌ "解决 collapsed and entangled":Pan 等 2024 已用此 framing
- ❌ "比 ensemble 更好":赛规禁止 ensemble,无可对比

---

## 11.6 立即更新到执行排期的两条结论

1. **W0 增加为独立准备周**(数据/环境/repo/账户认证/官方推理格式调研)
2. **创新模块拆三步推进**:Routing(W4) → Cross-Path Bilinear(W5) → Routing Sparsity Regularizer(W5 末),分阶段固化能跑出更可信的 ablation 表

