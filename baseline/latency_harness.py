"""Standalone inference-latency harness.

Loads a trained checkpoint and measures single-request (B=1) inference
latency. Useful for offline diagnosis of latency regressions, complementing
the post-training measurement that ``trainer.measure_inference_latency``
runs at the end of every ``train.py`` job.

Usage:
    python3 latency_harness.py \\
        --ckpt_dir /path/to/ckpt/global_step12345.layer=2.head=4.hidden=64.best_model \\
        --data_dir /path/to/parquet_dir \\
        [--num_warmup 50 --num_iters 1000 --use_bf16]

The harness expects ``schema.json`` and ``train_config.json`` to be present
inside ``ckpt_dir`` (sidecar files written by the trainer).
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

from utils import set_seed, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from train import build_feature_specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCVRHyFormer Inference Latency Harness")
    p.add_argument('--ckpt_dir', type=str, required=True,
                   help='Directory containing model.pt + schema.json + train_config.json')
    p.add_argument('--data_dir', type=str, default=None,
                   help='Parquet data dir (env: TRAIN_DATA_PATH). Used to construct one valid batch.')
    p.add_argument('--num_warmup', type=int, default=50)
    p.add_argument('--num_iters', type=int, default=1000)
    p.add_argument('--use_bf16', action='store_true', default=False,
                   help='Run forward in bfloat16 autocast.')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    return args


def main() -> None:
    args = parse_args()
    create_logger('/tmp/latency_harness.log')
    set_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    schema_path = ckpt_dir / 'schema.json'
    cfg_path = ckpt_dir / 'train_config.json'
    model_path = ckpt_dir / 'model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found at {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.json not found at {schema_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"train_config.json not found at {cfg_path}")

    with open(cfg_path, 'r') as f:
        train_cfg = json.load(f)
    logging.info(f"Loaded train_config from {cfg_path}")

    if not args.data_dir:
        raise ValueError("--data_dir is required (or set TRAIN_DATA_PATH)")

    seq_max_lens = {}
    for pair in train_cfg.get('seq_max_lens', '').split(','):
        if ':' not in pair:
            continue
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())

    _, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=str(schema_path),
        batch_size=int(train_cfg.get('batch_size', 256)),
        valid_ratio=float(train_cfg.get('valid_ratio', 0.1)),
        train_ratio=float(train_cfg.get('train_ratio', 1.0)),
        num_workers=2,
        buffer_batches=1,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    # NS groups: load from sidecar if present, else default singletons.
    ns_path = ckpt_dir / 'ns_groups.json'
    if ns_path.exists():
        with open(ns_path, 'r') as f:
            ns_cfg = json.load(f)
        u_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        i_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[u_idx[f] for f in fids] for fids in ns_cfg['user_ns_groups'].values()]
        item_ns_groups = [[i_idx[f] for f in fids] for fids in ns_cfg['item_ns_groups'].values()]
    else:
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=pcvr_dataset.user_dense_schema.total_dim,
        item_dense_dim=pcvr_dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=pcvr_dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        d_model=int(train_cfg['d_model']),
        emb_dim=int(train_cfg['emb_dim']),
        num_queries=int(train_cfg['num_queries']),
        num_hyformer_blocks=int(train_cfg['num_hyformer_blocks']),
        num_heads=int(train_cfg['num_heads']),
        seq_encoder_type=train_cfg['seq_encoder_type'],
        hidden_mult=int(train_cfg['hidden_mult']),
        dropout_rate=float(train_cfg['dropout_rate']),
        seq_top_k=int(train_cfg['seq_top_k']),
        seq_causal=bool(train_cfg['seq_causal']),
        action_num=int(train_cfg['action_num']),
        num_time_buckets=NUM_TIME_BUCKETS if train_cfg.get('use_time_buckets', True) else 0,
        rank_mixer_mode=train_cfg['rank_mixer_mode'],
        use_rope=bool(train_cfg.get('use_rope', False)),
        rope_base=float(train_cfg.get('rope_base', 10000.0)),
        emb_skip_threshold=int(train_cfg.get('emb_skip_threshold', 0)),
        seq_id_threshold=int(train_cfg.get('seq_id_threshold', 10000)),
        ns_tokenizer_type=train_cfg['ns_tokenizer_type'],
        user_ns_tokens=int(train_cfg.get('user_ns_tokens', 0)),
        item_ns_tokens=int(train_cfg.get('item_ns_tokens', 0)),
        include_ns_in_classifier=bool(train_cfg.get('include_ns_in_classifier', False)),
    ).to(args.device)
    state = torch.load(str(model_path), map_location=args.device)
    model.load_state_dict(state, strict=False)
    model.eval()

    logging.info(f"Loaded model from {model_path}")

    # Build a single batch then slice to B=1.
    first_batch = None
    for batch in valid_loader:
        first_batch = batch
        break
    if first_batch is None:
        raise RuntimeError("valid_loader yielded no batches")

    device_batch = {k: (v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in first_batch.items()}
    one_batch = {k: (v[:1].contiguous() if isinstance(v, torch.Tensor) else v)
                 for k, v in device_batch.items()}

    seq_domains = one_batch['_seq_domains']
    from model import ModelInput
    seq_data, seq_lens, seq_tb = {}, {}, {}
    for d in seq_domains:
        seq_data[d] = one_batch[d]
        seq_lens[d] = one_batch[f'{d}_len']
        B = one_batch[d].shape[0]
        L = one_batch[d].shape[2]
        seq_tb[d] = one_batch.get(f'{d}_time_bucket',
                                  torch.zeros(B, L, dtype=torch.long, device=args.device))
    model_input = ModelInput(
        user_int_feats=one_batch['user_int_feats'],
        item_int_feats=one_batch['item_int_feats'],
        user_dense_feats=one_batch['user_dense_feats'],
        item_dense_feats=one_batch['item_dense_feats'],
        seq_data=seq_data, seq_lens=seq_lens, seq_time_buckets=seq_tb,
    )

    use_cuda = (args.device.startswith('cuda')) and torch.cuda.is_available()

    def _autocast():
        if args.use_bf16 and use_cuda:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    with torch.no_grad():
        for _ in range(args.num_warmup):
            with _autocast():
                model.predict(model_input)
        if use_cuda:
            torch.cuda.synchronize()

        latencies_ms = []
        if use_cuda:
            for _ in range(args.num_iters):
                s_evt = torch.cuda.Event(enable_timing=True)
                e_evt = torch.cuda.Event(enable_timing=True)
                s_evt.record()
                with _autocast():
                    model.predict(model_input)
                e_evt.record()
                torch.cuda.synchronize()
                latencies_ms.append(float(s_evt.elapsed_time(e_evt)))
        else:
            for _ in range(args.num_iters):
                t0 = time.perf_counter()
                with _autocast():
                    model.predict(model_input)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms.sort()
    n = len(latencies_ms)
    print(f"=== Latency (B=1, n={n}, warmup={args.num_warmup}, "
          f"device={args.device}, bf16={args.use_bf16}) ===")
    print(f"  Mean: {sum(latencies_ms)/n:.3f} ms")
    print(f"  P50:  {latencies_ms[int(n*0.50)]:.3f} ms")
    print(f"  P95:  {latencies_ms[min(n-1, int(n*0.95))]:.3f} ms")
    print(f"  P99:  {latencies_ms[min(n-1, int(n*0.99))]:.3f} ms")
    print(f"  Min:  {latencies_ms[0]:.3f} ms / Max: {latencies_ms[-1]:.3f} ms")


if __name__ == "__main__":
    main()
