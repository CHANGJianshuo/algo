# W1 快赢 — 平台提交指引

## 〇、一次性上传(只做一次)

平台 Model Training → Create Training Job 页面,**先把所有非 run.sh 的文件上传一次**。Click `Upload from Local`,逐个上传以下 8 个文件(同名会覆盖,放心传):

| 文件 | 大小 | 说明 |
|---|---|---|
| `dataset.py` | 30K | 主办方原版,未改 |
| `model.py` | 65K | **已改**:加 `include_ns_in_classifier` 参数 |
| `train.py` | 22K | **已改**:加 W1 所有 CLI flag |
| `trainer.py` | 32K | **已改**:加 bf16 / EMA / lr scheduler / latency 测量 |
| `utils.py` | 11K | 主办方原版,未改 |
| `ns_groups.json` | 2K | 主办方原版,未改 |
| `ema.py` | 5K | **新增**:EMA 模型类 |
| `latency_harness.py` | 9K | **新增**:独立延迟测量(可选,本地用) |

> ⚠️ `run.sh` 在平台上**不能删除**(锁定的入口),只能 ✏ 编辑或上传同名覆盖。下面每个 Step 切换 run.sh 内容即可。

---

## 一、Step 0:基准 baseline_v0(必做,先跑这个)

不动 `run.sh`,直接 Submit。Job Name 填 `baseline_v0_unchanged`。

- **第一次跑加 `--num_epochs 1` 验证流程**:在 run.sh 末尾的 python3 命令最后加 `--num_epochs 1`,5-10 分钟出结果。**确认能跑通后**再去掉它正式跑。

---

## 二、各 Step 的 run.sh 完整内容

> 提交前:点平台 `run.sh` 那一行的 ✏ 图标,**清空**编辑器,**粘贴**对应 Step 块的内容,保存,改 Job Name,Submit。

### Step 1:`baseline_v1_bf16` — bfloat16

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_bf16 \
    "$@"
```

### Step 2:`baseline_v2_ema` — bf16 + EMA

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_bf16 \
    --use_ema --ema_decay 0.999 \
    "$@"
```

### Step 3:`baseline_v3_dropout` — bf16 + EMA + dropout 0.05

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_bf16 \
    --use_ema --ema_decay 0.999 \
    --dropout_rate 0.05 \
    "$@"
```

### Step 4:`baseline_v4_lrsched` — + LR warmup + cosine

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_bf16 \
    --use_ema --ema_decay 0.999 \
    --dropout_rate 0.05 \
    --use_lr_warmup --warmup_steps 1000 --lr_decay_steps 30000 --lr_min_ratio 0.1 \
    "$@"
```

### Step 5:`baseline_v5_nshead` — + NS 进分类头

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_bf16 \
    --use_ema --ema_decay 0.999 \
    --dropout_rate 0.05 \
    --use_lr_warmup --warmup_steps 1000 --lr_decay_steps 30000 --lr_min_ratio 0.1 \
    --include_ns_in_classifier \
    "$@"
```

### Step 6a:`baseline_v6a_focal` — A/B 切到 Focal Loss

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_bf16 \
    --use_ema --ema_decay 0.999 \
    --dropout_rate 0.05 \
    --use_lr_warmup --warmup_steps 1000 --lr_decay_steps 30000 --lr_min_ratio 0.1 \
    --include_ns_in_classifier \
    --loss_type focal --focal_alpha 0.1 --focal_gamma 2.0 \
    "$@"
```

> Step 6b 就是 Step 5(BCE),不需要再跑;直接拿 Step 5 的 AUC 当 6b 对照。

---

## 三、提交流程速查

每个 Step 走这 4 步:

1. **Job Name** 填上面对应名字(`baseline_v1_bf16` / `baseline_v2_ema` / ...)
2. **Job Description** 填这次改了什么(便于回看)
3. 点 `run.sh` 行 ✏ 编辑,清空,粘贴对应 Step 块,保存
4. **Resources** 填 `1`(单 Job 单卡,详见下文)
5. Submit

跑完 → Model Evaluation 出 AUC → 在 `EXPERIMENTS.md` 填表格那一行的 AUC 列 + log 里抓 `Post-training inference latency` 几行的 P50/P99。

---

## 四、关于 2 张 GPU 同时用

平台显示 `Total 2 / Available 2`。**有两种"用 2 张卡"的方式**:

### 方式 A:同时提交 2 个独立 Job(推荐,W1 现在就能用)

每个 Job 设 `Resources=1`,平台并行跑 → **两个 Step 同一时间出结果,W1 时间砍半**。

最佳搭配:
- **Job 1**(GPU1):跑 Step 5
- **Job 2**(GPU2):跑 Step 6a Focal
- 等两个都跑完,直接对比 5 vs 6a 的 AUC,定下 Step 6 的胜者

或者:
- 先跑 Step 1(bf16 验证速度)→ 等结果出来后
- 同时提交 Step 2(EMA)+ Step 3(dropout)→ 两个结果同时出
- 同时提交 Step 4(lr) + Step 5(NS) → 两个同时出
- 同时提交 Step 6a(focal) + 重新跑 Step 5(对照,如果数字波动)

### 方式 B:单 Job 用 2 GPU(DDP,**W1 不做**)

- 当前 baseline 是单进程 PyTorch,**没有 DDP/DataParallel**
- 简单调 `Resources=2` 不会自动并行 — 第二张卡空闲,白浪费资源额度
- 想真用:要给 trainer 接 DDP(改动 100-200 行),还要处理 sparse Adagrad 的多进程同步,**风险中等**
- **推荐放到 W2 后再考虑**(那时候模型变大、训练时间长,DDP 收益才明显)

### 我的建议

**W1 用方式 A**,Resources 永远填 1,两步并行提交。这样 7 步实际只需要 4-5 个时段(1 → [2,3] → [4,5] → [6a,5重跑] → 收尾),从 7 小时压到 3-4 小时。

---

## 五、提交清单 checklist

第一次提交前:

- [ ] 平台 Training Code 表格里 8 个文件都已上传(原 6 个 + ema.py + latency_harness.py;`run.sh` 平台已有不动)
- [ ] 运行 Step 0 baseline_v0 时 `run.sh` 是主办方原版(没动)
- [ ] 第一次跑加 `--num_epochs 1`,确认 5-10 分钟内出结果不报错
- [ ] EXPERIMENTS.md 已建好,Step 0 行的 AUC 已填(基准)

每个 Step N 提交前:

- [ ] Job Name 改成对应名字
- [ ] `run.sh` 内容已替换为对应 Step 块
- [ ] Resources = 1
- [ ] 上一步的 AUC 已记录在 EXPERIMENTS.md(用作对比基准)

跑完后:

- [ ] Model Evaluation 出 AUC,填进 EXPERIMENTS.md
- [ ] log 里抓 `Post-training inference latency` 的 P50/P99,填进 EXPERIMENTS.md
- [ ] AUC 提升 → 继续下一 Step;持平/下降 → 在备注里记原因

---

## 六、W2 KV-Aware Tokenization(核心创新点)

W1 微调对 emb-dominated 模型收益有限。**W2 直接改 NS Tokenizer 结构**,利用主办方故意留的 8 对 KV 对齐字段(`user_int_feats_{62-66,89-91}` ↔ `user_dense_feats_{62-66,89-91}`)。预期 +1~3 AUC pp。

### 上传清单(覆盖之前的)

`baseline/` 8 个文件(model.py / train.py 都改了),`evaluation/` 3 个文件(model.py 同步 + infer.py 改了)— 全部用最新版上传/覆盖。

### W2 提交版 run.sh(单独看 KV-Aware 净效果,不叠加 W1)

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type kv_aware \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --batch_size 128 \
    --use_bf16 \
    "$@"
```

**关键 flag**:
- `--ns_tokenizer_type kv_aware`(W2 核心,启用 KV-Aware)
- `--user_ns_tokens 5`(在 KV-Aware 下含义变成"非 KV 部分的 RankMixer token 数")
- 其他保持 baseline 默认
- `--use_bf16 --batch_size 128` 防显存爆(KV-Aware 多了一些参数)

Job Name 建议:`w2_kvaware_v1`

### 验证 log 头部

```
KVAwareUserNSTokenizer: 8/8 valid KV pairs (fids=[62, 63, 64, 65, 66, 89, 90, 91]),
5 non-KV RankMixer tokens over 38 fields, total 13 user NS tokens
PCVRHyFormer model created: num_ns=16, T=24, d_model=64, rank_mixer_mode=ffn_only
```

→ 看到 **`8/8 valid KV pairs`** 说明所有 8 对 KV 字段都被找到并启用。
→ `rank_mixer_mode=ffn_only` 是自动 fallback,正常(因为 d_model=64 不整除 T=24)。

### 评估失败排查

如果 evaluation 仍 fail,**必须**确认 `evaluation/` 下也上传了最新的 `model.py` 和 `infer.py`(它们都加了 KVAwareUserNSTokenizer 类和 user_int_fids/user_dense_field_specs 透传)。

### W2 后的 A/B(可选)

如果 KV-Aware 单跑提分(>baseline 0.8125),再叠加 W1 里**有把握**的项:

```bash
# W2 + EMA(完整版,不是 dense_only)
... --use_ema --ema_decay 0.999 ...

# W2 + warmup(短 warmup)
... --use_lr_warmup --warmup_steps 200 --lr_decay_steps 0 ...
```

每次只加 1 项,做 ablation。

---

## 七、Final Round 提示

平台说明:"During the Final Round, please upload a technical report (README.md in Markdown format) describing your approach."

→ **决赛要传一份 README.md 描述方案**。等 W7-W9 整理 PLAN.md 第八节时再写,现在不用管。
