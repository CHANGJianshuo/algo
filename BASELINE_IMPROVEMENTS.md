# Baseline 改进全维度方案

把 baseline(PCVRHyFormer)跟 `PLAN.md` 第十一节融在一起的**全维度改进方案**,从"几行代码"到"重写架构",按层组织。每条都标了 ⭐ROI 评级 + 理由 + 风险。

---

## 0. 评级图例

| 评级 | 含义 |
|---|---|
| ⭐⭐⭐⭐ | 必做,几乎零风险高收益 |
| ⭐⭐⭐ | 推荐,中风险中高收益 |
| ⭐⭐ | 进阶,需要实验验证 |
| ⭐ | 大改,高风险但创新奖必须 |
| 🚫 | 不能做(违规 / 陷阱) |

---

## 一、数据层

### 1.1 ⭐⭐⭐⭐ 时间分布检查 + Adversarial Validation
**现状**:`valid_ratio=0.1` 取 row group 尾部 10%(按时间最近)。
**改动**:训一个二分类器区分 train vs valid(用同一批特征),输出 AUC。
**为什么**:如果 AV-AUC > 0.55,说明分布漂移大,你优化的 valid AUC 不能反映 leaderboard;> 0.65 就要重新切分 / 加 domain alignment。
**风险**:如果不做,你可能在错的目标上爬山一周才发现。
**实施**:1 天,独立脚本,用 LightGBM 即可。

### 1.2 ⭐⭐⭐⭐ 多任务学习(Multi-Task)
**现状**:`label = (label_type == 2)`,丢掉了其他 label_type 的信号。
**改动**:同时预测 label_type ∈ {1, 2, ...}(点击 / 转化 / 加购等),共享 backbone 多 head。
**为什么**:label_type=1(点击)样本量大很多,作为辅助任务能让 backbone 学到更通用的表示;转化少导致主 loss 信号稀疏,辅助任务降方差。
**收益**:CTR/CVR 学术界共识 +0.5~1.5 AUC pp。
**实施**:`action_num` 已经支持多输出,改 trainer 加多 loss 即可。

### 1.3 ⭐⭐⭐ 序列采样增强(Sequence Augmentation)
**现状**:序列直接截最近 256/512 步。
**改动**:训练时随机 sample 60-100% 的 token,推理时全用。类似 BERT MLM 思路。
**为什么**:1M 样本 + 长序列容易过拟合到具体行为模式,采样能模拟"该用户的不同子轨迹",泛化提升。
**风险**:不能 mask 太狠(>40%),不然破坏时序;ID dropout 不能对低基数特征做。

### 1.4 ⭐⭐ Label Smoothing + 校准 loss
**现状**:硬 0/1 标签 + BCE。
**改动**:`y' = 0.95·y + 0.05·base_rate`,或者加 calibration loss 让 sigmoid 输出对齐真实 CVR。
**为什么**:推荐里 calibration 重要(预测要接近真实点击率),纯 BCE 倾向极端化。AUC 衡量排序但不衡量 calibration,加这个不一定提 AUC 但减泛化误差。

### 1.5 🚫 不要做
- 对 ID 做 Mixup(会变浮点 → 无意义)
- 测试集 pseudo-label(违规)
- 数据重采样(扭曲 calibration,反而伤 AUC)

---

## 二、特征 & Tokenization 层

### 2.1 ⭐⭐⭐⭐ KV-Aware Field Tokenization(PLAN 11.1 核心)

**现状**(baseline 最大弱点):
```python
user_dense (918 维) → Linear → [64 维] → 1 个 token
```
918 维一把全压成 1 个 token,信息严重压缩。

**根本问题**:8 个 dense 字段(`62/63/64/65/66/89/90/91`)与同号 user_int 是 **KV 对齐**的:

```
user_int_feats_62  = [类目101, 类目205, 类目307, ...]   ← Key
user_dense_feats_62 = [0.8,    0.3,     0.5,     ...]   ← Value(对应权重)
```

baseline 把它们**完全分开**处理 → 配对信息丢失。

**改动**:
1. 对 8 对 KV 字段:
   - 查 K 的 embedding ([N, 64])
   - 用 V 加权(weighted sum / softmax 加权)
   - → 每对产出 1 个 KV-aware token
2. 非 KV 的 dense(`61=SUM`, `87=LMF4Ads`)各 1 token
3. 总计:user dense 部分从 **1 token → 10 token**

**进一步**:做 **Value-driven Gated Pool**(PLAN 11.1 修正版):用 V 作为 gate,过滤掉低权重 K,避免 token 爆炸。

**为什么有效**:
- KV 是主办方刻意设计的"用户精细偏好画像",原方案 1 个 token 装不下
- 工业界 RecSys 文献(Alibaba MIMN/SIM)证明这种结构对 CVR 关键
- 这是创新奖故事线的硬抓手

**收益**:估计 +1~3 AUC pp,**单点最大改进**。
**风险**:增加 token 数会让 d_model%T 约束变严,需要重新调 d_model;V 加权的具体形式要 ablation(直接乘 / softmax / sigmoid gate)。

### 2.2 ⭐⭐⭐ Sequence Token 改成 Field-Aware
**现状**:每个时间步把 8-13 个字段 cat 后 Linear → 64 维。
**改动**:每个字段独立保留 64 维,在序列内做 **field-axis attention**(每个字段是一个独立 axis)。
**为什么**:cat + Linear 等价于无序的字段融合,attention 能让模型自己学"这个时间步主要看 item_id 还是 category"。
**收益**:序列建模质量提升,典型 +0.3~0.8 AUC。

### 2.3 ⭐⭐ Time2Vec / 连续时间编码
**现状**:Time bucket = 65 个离散 bin。
**改动**:用 Time2Vec(`sin(ωt + φ)` 多尺度)替代或拼接桶 embedding。
**为什么**:bucket 边界附近梯度跳变,连续编码更平滑;能表达"同一桶内的相对顺序"。

### 2.4 ⭐⭐⭐ 显式特征交叉(DCN-V2 风格)
**现状**:全靠 attention/MLP 隐式学交叉。
**改动**:在 NS token 之上加 **2 层 Cross Network**(DCN-V2),做显式二阶/三阶交叉。
**为什么**:推荐场景里"用户性别 × 商品类目"这种二阶交叉是核心信号,attention 学起来低效。
**注意**:不违反"单 block"约束 — Cross Network 是 NS tokenizer 的一部分。

---

## 三、架构层(改动越深风险越高)

### 3.1 ⭐⭐⭐⭐ 序列编码器换 HSTU(PLAN 11.2 核心)

**现状**:每个域独立 TransformerEncoder。
**改动**:替换为 HSTU(Meta 2024,Generative Recommenders 论文)。
**为什么**:
- HSTU 是专为推荐序列设计的,SOTA on industry-scale
- 比 Transformer 快 ~2×(线性 attention)
- 单 block 表达力 ≈ 多层 Transformer,契合赛题"单 block 架构"叙事
- Meta 已开源,有现成实现

**实施路径**:替换 `seq_encoders[i]`,接口完全兼容(都是 `[B, L, D] → [B, L, D]`)。
**收益**:典型 +0.5~1.5 AUC,加上速度优势可以训更多 epoch。

### 3.2 ⭐⭐ Domain Routing(MoE / Per-Token)
**现状**:4 个域 4 个独立 encoder,参数完全不共享。
**改动**:N 个共享 expert(N=8 比如),每个 token 动态选 K=2 个 expert(MoE)。
**为什么**:4 个域有共性也有特性,完全独立浪费参数,完全共享又抓不到差异。MoE 让模型自己决定共享什么。
**进阶**:**Per-Token Routing**(PLAN 11.5 真差异点)— 不是 sample 级,是 token 级,这是 baseline 完全没做的事。
**风险**:MoE 训练不稳,需要 load balance loss;1M 样本可能撑不起多 expert。

### 3.3 ⭐⭐⭐ Cross-Sequence Query Generation
**现状**:`MultiSeqQueryGenerator` 里每个域独立造 query,域间零信息流。
**改动**:让 seq_a 的 query 也能 attend seq_b/c/d 的摘要。
**为什么**:用户在 A 域的行为是 D 域的强 prior(同一用户跨域兴趣相关)。
**实施**:把 4 个域的 mean pool 拼成 [4, D],query gen 用 attention(query 是当前域,key/value 是 4 域)。

### 3.4 ⭐⭐⭐ NS Token 也进分类头
**现状**:`output_proj` 只用 query tokens 拼接,NS token 完全不进 final embedding!信息浪费。
**改动**:final pooling 同时用 query + NS,加权或 attention 聚合。
**为什么**:NS token 包含了所有静态画像,直接丢掉显然次优。
**实施**:就改 `_run_multi_seq_blocks` 最后几行,代码改动 < 30 行。
**收益**:小但稳,+0.1~0.3 AUC,几乎零风险。

### 3.5 ⭐ Unified Block(PLAN 创新核心,大改)
**现状**:每个 HyFormer Block 是 self-attn → cross-attn → mixer 三段串联。
**改动**:把三步合并为**单一统一操作**。具体:
- 把 query / NS / seq 当作一个统一 token 集(异构 attention)
- 用 token-level mask 区分类型
- 单次 attention + gating

**为什么**:
- 赛题字面要求"单 block 架构"
- 论文级别的故事:从 "decoupled" → "unified" 是创新点
- 推理加速明显(少一次 LayerNorm + 少一次 reshape)

**风险高**:训练不稳,需要 careful ablation。但成功了就是创新奖。

### 3.6 ⭐⭐ 加深(2 → 4 layer)
**现状**:`num_hyformer_blocks=2`。
**改动**:加到 4 或 6,加 stochastic depth(随机跳层)。
**为什么**:1M 样本能撑 30M 参数,baseline 太浅了。
**注意**:同时加正则,不然过拟合。

---

## 四、训练范式

### 4.1 ⭐⭐⭐⭐ EMA(指数移动平均)
**现状**:无。
**改动**:维护 EMA 模型权重(decay=0.999),用 EMA 做 inference。
**为什么**:
- **EMA 是赛题里唯一合法的"伪 ensemble"!**
- 几乎零成本,代码 30 行
- 几乎所有 SOTA 模型都用,典型 +0.3~0.8 AUC

**实施**:每个 step 后 `ema_w = 0.999*ema_w + 0.001*model_w`。

### 4.2 ⭐⭐⭐⭐ bfloat16 训练
**现状**:float32。
**改动**:`torch.cuda.amp` + bfloat16 autocast。
**为什么**:A100 上速度 ×1.5-2,显存 ×0.5,精度无损。**必做**。
**注意**:Adagrad 优化器对 amp 兼容性差,用 fp32 master weight。

### 4.3 ⭐⭐⭐ 学习率 Warmup + Cosine Decay
**现状**:固定 lr=1e-4。
**改动**:warmup 1000 step → cosine decay 到 1e-5。
**为什么**:CTR 模型在 lr schedule 上敏感,固定 lr 容易在中后期震荡。
**收益**:典型 +0.2~0.5 AUC。

### 4.4 ⭐⭐⭐ 自监督预训练(Self-Supervised Pretraining)
**现状**:无,直接监督学习。
**改动**:先在 1M 样本上做 **Masked Sequence Modeling** 预训练 → finetune 到 CVR。
**为什么**:
- 1M 样本对 30M 参数模型来说是少数据情景
- BERT/SimCLR 风格预训练在 RecSys 已被验证
- 可以加进创新奖故事

**陷阱**:**预训练时间不能穿越**(PLAN 11.4)— 不能用未来 mask 预测过去。

### 4.5 ⭐⭐ R-Drop
**现状**:无。
**改动**:同一 batch 前向两次(两次不同 dropout),对两次输出加 KL loss。
**为什么**:正则化效果显著,代码简单,RecSys 上多次验证有效。

### 4.6 ⭐⭐ 大 batch + Gradient Accumulation
**现状**:batch=256。
**改动**:grad accum 到等效 1024 / 2048。
**为什么**:CTR 任务正样本少,大 batch 让每个 batch 包含更多正样本,梯度更稳。
**注意**:相应调高 lr(linear scaling rule)。

---

## 五、损失函数

### 5.1 ⭐⭐⭐ Focal Loss A/B(baseline 已支持)
**改动**:开 `--loss_type focal --focal_alpha 0.1 --focal_gamma 2`。
**为什么**:CVR 正样本极少(1-3%),Focal Loss 让模型聚焦难样本。
**注意**:并非总是好 — 要 A/B 测试。有时候 BCE+pos_weight 更稳。

### 5.2 ⭐⭐⭐ 辅助损失(PLAN 第二节 3 个)
- **Click 任务**:同时预测 label_type=1
- **Sequence MLM**:masked next-item prediction
- **Item Popularity**:预测 item 在序列里的位置

**为什么**:多个 loss 共享 backbone,正则化 + 信号增强。
**实施**:加权和,权重需调(典型 [1.0, 0.3, 0.1])。

### 5.3 ⭐⭐ Calibration Loss
**改动**:加一项 `(mean(sigmoid) - true_pos_rate)²` 到总 loss。
**为什么**:校准好的概率在线上 ranking 更稳定(虽然 AUC 看不出区别)。

---

## 六、正则化

### 6.1 ⭐⭐⭐ 调高 dropout
**现状**:`dropout_rate=0.01`(很低)。
**改动**:试 0.05-0.1。
**为什么**:推荐里 emb 大、容易过拟合,baseline 这个 dropout 偏低。

### 6.2 ⭐⭐ Frequency-Aware L2
**改动**:对不同频次的 ID embedding 用不同 L2 强度(高频小,低频大)。
**为什么**:低频 ID 容易记噪声,需要更强正则。

### 6.3 ⭐⭐ Stochastic Depth
**改动**:训练时随机跳过 HyFormer block(prob=0.1)。
**为什么**:加深模型时配套使用,避免深层过拟合。

---

## 七、推理 / 延迟(硬约束!)

### 7.1 ⭐⭐⭐⭐ Latency Harness(PLAN 第三节)
**必须 W0 就建立**:本地用 batch=1 测真实推理延迟(模拟线上),建一个 CI 脚本每次 commit 跑。
**为什么**:**延迟超标直接 DQ**,不能等到提交才发现。

### 7.2 ⭐⭐⭐ torch.compile
**改动**:`model = torch.compile(model)`。
**为什么**:PyTorch 2.x 默认提速 1.5-2×,无精度损失,改 1 行。
**注意**:首次编译慢,后续推理快。

### 7.3 ⭐⭐⭐ FlashAttention
**改动**:用 `nn.functional.scaled_dot_product_attention`(自动用 FA2)。
**为什么**:序列长度 256/512,attention 是延迟大头,FA 提速明显。

### 7.4 ⭐⭐ INT8 量化(PTQ)
**改动**:训练完做 Post-Training Quantization。
**为什么**:推理速度 ×2,精度损失 < 0.1 AUC。
**风险**:Embedding 量化要谨慎,可能伤精度。

### 7.5 ⭐ 知识蒸馏(终极手段)
**改动**:训一个大模型当 teacher,蒸馏到小模型。
**为什么**:大 model 训得好,但延迟超标 → 蒸馏到小 model 保留大部分性能。
**风险**:不能 ensemble,但**蒸馏不算 ensemble**(单模型推理)。

---

## 八、创新奖专用(PLAN 11.5)

这部分不一定提 AUC 多少,但**决赛汇报必须有**:

### 8.1 ⭐ Per-Token Routing
不是 sample 级,**每个序列 token 根据"距今多久"动态选不同 expert**。Recency Routing。
**故事**:把"时间衰减"显式建模成 routing 信号。

### 8.2 ⭐ Routing Sparsity ↔ Scaling Law
4 维 sweep:depth × width × seq_len × **sparsity**。前 3 个是常规,加上 sparsity 是差异化。
**故事**:在 fixed FLOPs 预算下,sparsity 是最优 frontier 上的关键变量。

### 8.3 ⭐ KV-Aware Tokenization
2.1 详述。这是数据层创新。
**故事**:充分利用赛题数据的 KV 对齐结构,baseline 没用。

### 8.4 ⭐ Unified Block
3.5 详述。架构层创新。
**故事**:从 decoupled 到 unified,符合赛题"单 block"导向。

---

## 九、🚫 必须避开的坑

| 不要做 | 后果 |
|---|---|
| Ensemble / Stacking | DQ |
| TTA(测试增强) | 延迟超标 DQ |
| Test set pseudo-label | 违规 |
| ID Mixup | 数值变浮点,无意义 |
| 序列里有未来时间 | 时间穿越,LB 大跌 |
| 预训练 emb 用全数据(包括 valid) | 数据泄漏 |
| 单纯加 d_model 不调 dropout | 过拟合,LB 跌 |
| 只看 valid AUC 不看 LB | valid 可能不代表 test 分布 |

---

## 十、实施路线图(按 ROI 排序)

| 周 | 工作 | 预期累计提升 |
|---|---|---|
| **W0** | 提交 baseline 拿基准 + Latency Harness + AV 检查 | 基准建立 |
| **W1** | bfloat16 + EMA + LR warmup + Focal A/B + dropout 调高 | +1.5~3 AUC pp |
| **W2** | **KV-Aware Tokenization**(PLAN 11.1) | +1~3 AUC pp |
| **W3** | Multi-Task(主+点击+辅助) + R-Drop | +0.5~1.5 |
| **W4** | **HSTU 替换** + Cross-Sequence Query Gen | +0.5~1.5 |
| **W5** | NS token 进分类头 + Field-Aware Seq + DCN-V2 cross | +0.3~1 |
| **W6** | **Self-Supervised Pretraining** | +0.3~1 |
| **W7** | **Unified Block** + Per-Token Routing(创新核心) | +0/-0.5(看创新效果) |
| **W8** | Scaling Law sweep + 推理优化决战 | 延迟达标 |
| **W9** | 创新奖材料整理 | 决赛准备 |

---

## 十一、几条原则

1. **每改一个东西就提一次** — baseline 先跑一次拿基准,然后每改 1-2 个超参/组件提一次,记录 EXPERIMENTS.md。**绝对不要憋大招**。
2. **改动按"代码改动量 × 预期收益"排序** — KV-Aware 改动小但收益高,Unified Block 改动大且不一定立刻见效,前者先做。
3. **Ablation 必做** — 加了 EMA,要能告诉自己"是 EMA 提升了 X AUC,还是改的别的"。否则后期不知道哪个 work。
4. **保持单模型 + 单 forward** — 别为了刷 AUC 偷偷加 ensemble,DQ 不可逆。
5. **创新奖 ≠ 高 AUC** — 决赛展示要有"为什么我的方法 novel" 的故事,2.1 + 3.5 + 8.x 是必备弹药。
