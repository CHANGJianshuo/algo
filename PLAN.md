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
