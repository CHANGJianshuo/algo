# Baseline 架构详解(PCVRHyFormer)

主办方 baseline 完整架构通俗解读。基于 `baseline/` 下 7 个文件(`dataset.py` / `model.py` / `trainer.py` / `train.py` / `utils.py` / `run.sh` / `ns_groups.json`)。

---

## 一、最重要的一张图(全局)

```
┌────────────── 输入 ──────────────┐
│ user_int (整数特征,~100 维)      │
│ user_dense (918 维浮点)          │
│ item_int  (整数特征,~30 维)      │
│ seq_a/b/c/d (4 个域,变长)        │
│ time_buckets (每步距今多久)       │
└──────────────────────────────────┘
                 │
    ┌────────────┴───────────────┐
    │ ① 特征 → Token 化           │
    └────────────┬───────────────┘
                 │  (16 个 NS Token + 4 套 Sequence Token)
    ┌────────────┴───────────────┐
    │ ② Query Generator           │  ← 每个序列域单独"造 Query"
    └────────────┬───────────────┘
                 │  (8 个 Query Token = 4 域 × 2 query)
    ┌────────────┴───────────────┐
    │ ③ HyFormer Block ×2  循环2次:│
    │   a. Sequence Evolution    │  ← Self-Attention 在序列内
    │   b. Query Decoding        │  ← Cross-Attention 从序列读信息
    │   c. RankMixer 融合        │  ← MLP-Mixer 在 token 间混合
    └────────────┬───────────────┘
                 │  (输出 8 个 Query Token,每个 64 维)
    ┌────────────┴───────────────┐
    │ ④ Output Projection         │  ← 拼起来 → Linear → 64 维
    └────────────┬───────────────┘
                 │  (1 个 64 维向量代表整个样本)
    ┌────────────┴───────────────┐
    │ ⑤ Classifier (MLP)          │
    └────────────┬───────────────┘
                 ▼
            转化概率 logit
```

**关键数字**:
- T = 16 个 NS Token + 8 个 Query Token = **24 个 token 在 attention 里**
- 每个 token = **64 维**
- HyFormer Block 堆 **2 层**

---

## 二、阶段 ①:特征 → Token 化

### 2.1 NS Tokenizer(用户/商品的"静态特征"打包成 token)

baseline 默认用 **RankMixerNSTokenizer**(模仿 RankMixer 论文)。工作流程:

```
Step 1:每个特征字段独立查 Embedding 表
─────────────────────────────────────
user_int_feats_1  (vocab=10)    →  Embedding(10, 64)  →  [64 维]
user_int_feats_3  (vocab=200)   →  Embedding(200, 64) →  [64 维]
user_int_feats_60 (数组,vocab=50K) →  Embedding(50K, 64) →  数组每个 ID 查表 + mean pool → [64 维]
...46 个 user_int 字段全部查表后...

Step 2:全部 64 维向量首尾相接拼成大长条
─────────────────────────────────────
[64, 64, 64, 64, ..., 64]  ← 拼成 [46 × 64] = 2944 维大向量

Step 3:切成 5 段(user_ns_tokens=5)
─────────────────────────────────────
[2944] → split → [589, 589, 589, 589, 588]
                     ↓        ↓        ↓        ↓        ↓
                  Linear  Linear  Linear  Linear  Linear     ← 各自独立投影到 64 维
                     ↓        ↓        ↓        ↓        ↓
                  [64]     [64]     [64]     [64]     [64]

Step 4:得到 5 个 user NS token
─────────────────────────────────────
user_ns_tokens = [B, 5, 64]
```

> 💡 **这是种"暴力切分"**:不关心哪些字段语义相近,直接按维度均分。这就是为什么 PLAN.md 11.1 要做 **KV-Aware Field Tokenization** — 按字段语义和 KV 对齐结构来切,信息保留更好。

**item 侧同理**,产生 **2 个 item NS token**(`item_ns_tokens=2`)。

### 2.2 Dense 特征 Token 化(简单粗暴)

```
user_dense (918 维) → Linear → LayerNorm → SiLU → [64 维] → 1 个 token
item_dense (空)    → 0 个 token
```

**全部 NS Token 加起来:5 + 1 + 2 + 0 = 8 个**

### 2.3 序列 Token 化(每个时间步变成一个 token)

序列数据是 `[B, S 字段, L 时间步]`。每个时间步要先变成一个 64 维 token:

```
seq_a 在某个时间步 t:
  字段 1 ID = 5    →  Embedding → [64]
  字段 2 ID = 132  →  Embedding → [64]
  ...
  字段 8 ID = 87   →  Embedding → [64]

→ 把 8 个 [64] 拼成 [8×64=512]
→ 过 Linear → [64]
→ 加上 time_bucket embedding(这一步距今多久) → [64]
→ 这就是这个时间步的 token
```

整个 seq_a 输出 `[B, 256, 64]`(B 个样本,256 个时间步,每个 64 维)。
seq_b/c/d 同理,长度分别是 256/512/512。

> 💡 注意 baseline 这里用 **拼接 + Linear**(信息保留好),而 user_dense 用 **直接 Linear**(信息损失大)。这种不对称是 baseline 的弱点。

---

## 三、阶段 ②:Query Generator(给每个序列"造 Query")

baseline 一个**容易忽视但很关键**的设计,在 `MultiSeqQueryGenerator` 里。

### 类比:把它想成"提问"

如果直接用 self-attention 处理一切,模型不知道"我究竟要预测什么"。所以 baseline 做了一个**显式的 query 生成**:

```
对每个序列域 i(共 4 个),独立生成 num_queries=2 个 query token:

Step 1:汇总所有"上下文"
   global_info_i = Concat(
     所有 NS token 展平,         ← user/item 静态信息
     MeanPool(seq_i)             ← 第 i 个序列的均值摘要
   )                              [shape: (8+1) × 64 = 576 维]

Step 2:为这个序列做 2 次"问问题"
   query_i_1 = FFN_{i,1}(global_info_i)  → [64 维]
   query_i_2 = FFN_{i,2}(global_info_i)  → [64 维]

   ※ 每个 (序列, query) 都有自己的独立 FFN,不共享!
```

**意思**:每个 query 都是"看着 user 资料 + item 资料 + 这个序列的整体气氛"提出来的"问题",然后下一阶段会拿这个问题去**精读**该序列。

→ **总共 4 序列 × 2 query = 8 个 query token**。

---

## 四、阶段 ③:HyFormer Block(主干循环 2 次)

每个 block 做 3 件事:

### 4.1 Sequence Evolution(序列内部演化)

每个序列**独立**过一个 Self-Attention 编码器(默认 transformer 模式):

```
seq_a [B, 256, 64] ──→ TransformerEncoder ──→ [B, 256, 64] (更新后的)
seq_b [B, 256, 64] ──→ TransformerEncoder ──→ [B, 256, 64]
seq_c [B, 512, 64] ──→ TransformerEncoder ──→ [B, 512, 64]
seq_d [B, 512, 64] ──→ TransformerEncoder ──→ [B, 512, 64]
```

四个序列**各有自己的 encoder**(参数不共享)。

> 💡 PLAN.md 第十一节的 **Domain-Routed Gated HSTU** 想法:这里其实没有真正的"路由",只是 4 个并列的独立 encoder。可以加上"基于域 ID 的 routing 门控",或者把 encoder 换成 HSTU。

### 4.2 Query Decoding(Query 从序列里读信息)

每个序列的 query 用 **Cross-Attention** 从该序列里抽取信息:

```
对序列 i:
  Q = q_tokens_list[i]   (B, 2, 64)   ← 阶段②造的 query
  K = V = next_seq_i      (B, L, 64)   ← 4.1 演化后的序列

  cross_attn(Q, K, V) →  (B, 2, 64)   ← query 被"填充"了序列细节信息
```

**类比**:Query 是问题,Sequence 是文档,Cross-Attention 是"从文档里找答案写到问题上"。

### 4.3 RankMixer 融合(所有 token 互相混合)

把所有 query 和 NS token **拼成一个 token 集合**:

```
combined = [
  q_a_1, q_a_2,        ← seq_a 的 2 个 query
  q_b_1, q_b_2,        ← seq_b 的 2 个 query
  q_c_1, q_c_2,        ← seq_c 的 2 个 query
  q_d_1, q_d_2,        ← seq_d 的 2 个 query
  ns_1, ns_2, ..., ns_8 ← 8 个 NS token
]
shape: (B, 8 + 8 = 16, 64)  ← 16 个 token,每个 64 维
```

→ 过 **RankMixerBlock**(MLP-Mixer 风格):
- **Token Mixing**:把矩阵转置,在 token 维度做 Linear(16 个 token 互相交换信息)
- **Per-Token FFN**:每个 token 独立过 MLP

要求 `d_model % T == 0`(64 % 16 = 0 ✓)。

→ 输出还是 `(B, 16, 64)`,然后**拆回**:前 8 个是更新后的 query,后 8 个是更新后的 NS。

### 4.4 第二个 HyFormer Block

把 4.1~4.3 再做一遍,query 和 NS 进一步精炼。

---

## 五、阶段 ④ + ⑤:输出与分类

```
最终输出的 8 个 query token(经过 2 层 HyFormer 之后):
  shape: (B, 8, 64)
       ↓
  flatten → (B, 8 × 64 = 512)
       ↓
  Linear(512 → 64) + LayerNorm → (B, 64)   ← 这是整个样本的"总结向量"
       ↓
  Classifier:
    Linear(64→64) + LN + SiLU + Dropout + Linear(64→1)
       ↓
  logit (B, 1)
       ↓ sigmoid
  转化概率 (B, 1)
```

⚠️ **NS token 没有进最终输出!** 只有 8 个 query 进了分类头。NS token 只在 HyFormer 内部参与混合,起"信息源"作用。

---

## 六、再画一遍信息流(简化版)

```
特征
 │
 ├─ 静态特征 ──────► NS Tokens ────────────────┐
 │  (user/item)                                 │
 │                                              ▼
 │                                     ┌───────────────┐
 ├─ 序列(4 域)──► Seq Tokens ──┐      │   每个 block: │
 │                                │      │               │
 │                                ▼      │  ┌────────┐  │
 │            生成 ──► Query   ──►Cross─►│  │RankMixer│ │
 │                                Attn   │  │(token   │ │
 │                                  ▲    │  │ mixing) │ │
 │  序列 ◄─── Self-Attention ──────┘     │  └────────┘  │
 │                                       └───────────────┘
 │                                              │
 │                                              ▼
 │                                       Query 被多次精炼
 │                                              │
 │                                              ▼
 └─────────────────────────────► 拼接 + Linear + MLP → 转化概率
```

---

## 七、用一句话概括每个组件

| 组件 | 干啥的 | 类比 |
|---|---|---|
| **NS Tokenizer** | 把几十个静态特征打包成 8 个 token | 把几十张身份证压缩成 8 张摘要卡 |
| **Sequence Embedding** | 每个时间步的几个 ID → 一个 token | 把每条历史行为编码成一句话 |
| **Time Bucket Embedding** | 给每步行为标"距今多久"标签 | 在时间线上加刻度 |
| **Query Generator** | 为每个序列造 2 个"提问" | 站在问题角度先想好要问什么 |
| **Sequence Encoder (Transformer)** | 序列内部 token 互看 | 重新理解每条历史 |
| **Cross-Attention** | Query 从序列里取答案 | 拿着问题翻文档 |
| **RankMixer** | Query + NS 互相混合 | 不同视角的答案互相印证 |
| **Classifier** | Query → 概率 | 综合所有答案下结论 |

---

## 八、几个不容易想到的设计细节

| 细节 | 说明 |
|---|---|
| **NS token 不进 attention,只进 RankMixer** | NS 是"上下文",query 才是"主角" |
| **每个序列有独立 encoder/cross-attn** | 4 域差异大,不共享参数 |
| **Query 由"NS + Seq 摘要"显式构造** | 而不是 learnable 随机初始化 |
| **每个时间步有 time_bucket embedding** | 加权"近期 vs 远期"信号 |
| **`emb_skip_threshold=1M`** | vocab > 100 万的特征**直接用零向量代替**(省显存,但损失信息) |
| **`seq_id_threshold=10K`** | vocab > 10K 的字段在 sequence 里训练时多一倍 dropout(反过拟合) |
| **`reinit_sparse_after_epoch=1`** | 第二个 epoch 起,每 epoch 末尾重置高基数 embedding(KuaiShou 的 MultiEpoch trick) |

---

## 九、认知坐标(对应 PLAN.md 改进点)

读完这个 baseline,可以总结:

1. **它本质上是 "Decoder 风格的 Transformer + 推荐系统改造"**
   - Query Decoder 思路来自 DETR/Perceiver
   - Token 化的目标特征不是"序列",是"字段集合"

2. **创新空间集中在 5 个地方**:
   - **NS Tokenizer**:暴力均分太粗暴 → KV-Aware Field Tokenization (PLAN 11.1)
   - **Sequence Encoder**:4 个独立 transformer → HSTU + Domain Routing (PLAN 11.2)
   - **Query Generator**:简单 FFN 造 query → 可以更结构化
   - **HyFormer Block**:每层都 self-attn + cross-attn + mixer 太重 → unified block
   - **Classifier**:只用 query 不用 NS → 信息浪费

3. **它**不是简单 baseline**,是个完整的 SOTA 候选**。改赢它需要真本事 — 但也意味着**改动 1-2 个组件就可能见效**,不需要重写一切。

---

## 十、训练时长粗估

| 项目 | 数值 |
|---|---|
| 每 epoch step 数 | ~3500(训练 900K / batch 256) |
| 每 step 耗时 | 50-100 ms(瓶颈在 parquet IO) |
| 每 epoch | 5-7 分钟 |
| 收敛 epoch | 3-8 |
| Patience | 5 |
| **总训练时长** | **40 分钟 - 1.5 小时,典型 1 小时** |
| 端到端(含排队) | **1-2 小时** |
