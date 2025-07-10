#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# === Configuration ===
SRC=en
TGT=vi
DATA_DIR=$SCRIPT_DIR/kgnmt_data
BIN_DATA_DIR=$SCRIPT_DIR/data-bin/envi-bpe
CHECKPOINT_DIR=$SCRIPT_DIR/checkpoints/envi-dynamic-kgnmt

OUT_DIR=$SCRIPT_DIR/output/all_checkpoints
SPLIT="test"  # can change to valid, train, etc.

mkdir -p "$OUT_DIR"

# === Iterate over all checkpoints ===
echo "Evaluating all checkpoints in $CHECKPOINT_DIR..."
for ckpt in "$CHECKPOINT_DIR"/checkpoint*.pt; do
  ckpt_name=$(basename "$ckpt" .pt)
  result_dir="$OUT_DIR/$ckpt_name"
  mkdir -p "$result_dir"
  
  echo "Generating with checkpoint: $ckpt_name"

  fairseq-generate "$BIN_DATA_DIR" \
    --task translation_dynamic_knowledge_aug \
    --arch dynamic_kgnmt_iwslt_vi_en \
    --path "$ckpt" \
    --batch-size 128 --beam 5 --remove-bpe \
    --source-lang "$SRC" --target-lang "$TGT" \
    --results-path "$result_dir" \
    --gen-subset "$SPLIT"
  
  echo "Done: $ckpt_name → $result_dir"
done

echo "All checkpoint evaluations complete. Results saved in: $OUT_DIR"
