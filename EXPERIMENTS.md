# 实验记录

每次提交平台训练 Job 在这里加一行。**先有基准,才能谈改进**。

## 提交指引

平台入口固定 `bash run.sh`。每个 Step 提交时,把 `baseline/run_quickwin.sh` 的对应 toggle 取消注释,**临时**覆盖到 `run.sh`,打包 `baseline/` 上传。提交完成后可以恢复 `run.sh`。

```bash
# Step N 提交流程(以 Step 1 为例)
cp baseline/run_quickwin.sh baseline/run.sh   # 临时覆盖
# 在 run.sh 里取消 --use_bf16 那一行的注释
# 打包 baseline/ 上传到平台
# 提交完成后:
git checkout baseline/run.sh                  # 恢复主办方原版
```

每次提交在平台 Model Evaluation 出 AUC 后,在下方表格 `验证 AUC` 列填数。

## W1 快赢实验

| # | Job Name | 改动 / 加的 flag | 参数量 | epoch 用时 | 早停 epoch | 验证 AUC | 推理 P50 | 推理 P99 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline_v0 | (无,原版 run.sh) | TBD | TBD | TBD | **TBD** | TBD | TBD | 基准线 |
| 1 | baseline_v1_bf16 | `--use_bf16` | TBD | TBD | TBD | **TBD** | TBD | TBD | 速度应 ↑ 1.5-2× |
| 2 | baseline_v2_ema | + `--use_ema --ema_decay 0.999` | TBD | TBD | TBD | **TBD** | TBD | TBD | 期望 +0.3~0.8 AUC |
| 3 | baseline_v3_dropout | + `--dropout_rate 0.05` | TBD | TBD | TBD | **TBD** | TBD | TBD | 正则加强 |
| 4 | baseline_v4_lrsched | + `--use_lr_warmup --warmup_steps 1000 --lr_decay_steps 30000 --lr_min_ratio 0.1` | TBD | TBD | TBD | **TBD** | TBD | TBD | warmup+cosine |
| 5 | baseline_v5_nshead | + `--include_ns_in_classifier` | TBD | TBD | TBD | **TBD** | TBD | TBD | output_proj 维度变 |
| 6a | baseline_v6_focal | + `--loss_type focal --focal_alpha 0.1 --focal_gamma 2.0` | TBD | TBD | TBD | **TBD** | TBD | TBD | A/B 测试 |
| 6b | baseline_v6_bce | (保留 BCE,即 v5 状态) | TBD | TBD | TBD | **TBD** | TBD | TBD | A/B 对照 |

## 注

- **Step 7 (Latency Harness)** 不需要单独提交 — 它由 `trainer.train()` 末尾自动触发,每个 Step 跑完都会输出 P50/P95/P99 到 log,填到上面 `推理 P50 / 推理 P99` 列即可。
- 用 `--num_epochs 1` 做第一次"流程跑通"验证(5-10 分钟出结果,不计入实验 AUC)。
- 真实实验不带 `--num_epochs 1`,让早停决定。
- 每次提交后从 log 的 `Total parameters: ...` / `Epoch X Validation` / `Post-training inference latency` 几行抓数。

## 决策

- `Step 6` 选 AUC 高的那一版作为 W2 起点。
- 所有提升保留;持平/下降的项,在备注里写原因(超参不对 / 实现 bug / 数据漂移)再回滚。
