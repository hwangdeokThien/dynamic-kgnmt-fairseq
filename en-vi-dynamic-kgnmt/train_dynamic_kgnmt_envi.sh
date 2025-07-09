#!/usr/bin/env bash

# thesis - Thien and Truyen

set -e

# === Resolve script directory ===
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# === Configuration ===
SRC=en
TGT=vi
VOCAB_SIZE=12000
MODEL_PREFIX=$SCRIPT_DIR/tokenizer/envi
DATA_DIR=$SCRIPT_DIR/kgnmt_data
BIN_DIR=$SCRIPT_DIR/data-bin/envi-bpe
SP_MODEL=$MODEL_PREFIX.model
CHECKPOINT_DIR=$SCRIPT_DIR/checkpoints/envi-dynamic-kgnmt
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
CUDA_VISIBLE_DEVICES=0 fairseq-train "$BIN_DIR" \
  --distributed-world-size 1 \
  --ddp-backend no_c10d \
  --task translation_dynamic_knowledge_aug \
  --arch dynamic_kgnmt_iwslt_vi_en \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 4096 \
  --max-epoch 60 \
  --max-update 100000 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu \
  --maximize-best-checkpoint-metric \
  --save-dir "$CHECKPOINT_DIR" \
  --encoder_embed_dim 512 \
  --encoder_layers 6 \
  --decoder_embed_dim 512 \
  --decoder_layers 6 \
  --dropout 0.3 \
  --share_decoder_input_output_embed \
  --activation_fn relu \
  --attention_dropout 0.0 \
  --activation_dropout 0.0 \
  --sample_times 5 \
  --src_encoder_embed_dim 512 \
  --src_encoder_layers 6 \
  --knw_encoder_embed_dim 512 \
  --knw_encoder_layers 6 \
  --knowledge_selector_dropout 0.1 \
  --knowledge_selector_attention_dropout 0.0 \
  --knowledge_selector_activation_dropout 0.0 \
  --knowledge_selector_activation_fn relu \
  --kgnmt-lr 0.0005 \
  --kgnmt-optimizer adam \
  --kgnmt-adam-betas '(0.9, 0.98)' \
  --kgnmt-weight-decay 0.0001 \
  --kgnmt-lr-scheduler inverse_sqrt \
  --kgnmt-warmup-updates 4000 \
  --kgnmt-warmup-init-lr -1.0 \
  --knowledge-selector-lr 0.0001 \
  --knowledge-selector-optimizer adam \
  --knowledge-selector-adam-betas '(0.9, 0.98)' \
  --knowledge-selector-weight-decay 0.0001 \
  --knowledge-selector-lr-scheduler polynomial_decay \
  --knowledge-selector-warmup-updates 1000 \
  # --share-decoder-input-output-embed \
  # --knowledge-selector-warmup-init-lr -1.0 

echo "✅ Training complete! Checkpoints saved in: $CHECKPOINT_DIR"