#!/bin/bash
# millow_5: DCN-V2 cross network on NS tokens + W2 KV sum-pool +
# extended sequences + full EMA. See ../README.md for design rationale.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type kv_aware \
    --user_ns_tokens 5 --item_ns_tokens 2 \
    --num_queries 2 --ns_groups_json "" \
    --emb_skip_threshold 1000000 --num_workers 8 \
    --batch_size 64 \
    --use_bf16 \
    --use_ema --ema_decay 0.999 \
    --use_dcn --dcn_layers 2 \
    --seq_max_lens seq_a:512,seq_b:512,seq_c:512,seq_d:1024 \
    --seq_encoder_type longer --seq_top_k 200 \
    "$@"
