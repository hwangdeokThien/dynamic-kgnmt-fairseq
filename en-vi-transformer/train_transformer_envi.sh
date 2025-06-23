#!/usr/bin/env bash

# thesis - Thien and Truyen

set -e

# === Resolve script directory ===
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# === Configuration ===
SRC=en
TGT=vi
VOCAB_SIZE=8000
MODEL_PREFIX=$SCRIPT_DIR/tokenizer/envi
DATA_DIR=$SCRIPT_DIR/trans_data
BIN_DIR=$SCRIPT_DIR/data-bin/envi-bpe
SP_MODEL=$MODEL_PREFIX.model
NUM_WORKERS=4
CHECKPOINT_DIR=$SCRIPT_DIR/checkpoints/envi-transformer
CHECKPOINT_FILE=$CHECKPOINT_DIR/checkpoint_latest.pt  # Path to the latest checkpoint

# === Create checkpoint directory if it doesn't exist ===
mkdir -p "$CHECKPOINT_DIR"

# === Check if checkpoint exists ===
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoint found! Resuming training from $CHECKPOINT_FILE"
else
    echo "No checkpoint found. Starting training from scratch."
fi

# === Train the model ===
# CUDA_VISIBLE_DEVICES=0,1 fairseq-train "$BIN_DIR" \
#   --distributed-world-size 2 \
fairseq-train "$BIN_DIR" \
  --ddp-backend no_c10d \
  --arch transformer_iwslt_vi_en \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr 5e-4 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --dropout 0.3 \
  --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 4096 \
  --max-epoch 50 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu \
  --maximize-best-checkpoint-metric \
  --save-dir "$CHECKPOINT_DIR" \
  # --restore-file "$CHECKPOINT_FILE"

echo "✅ Training complete! Checkpoints saved in: $CHECKPOINT_DIR"
