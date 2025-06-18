#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# === Configuration ===
SRC=en
TGT=vi
DATA_DIR=$SCRIPT_DIR/trans_data
SPM_DATA_DIR=$SCRIPT_DIR/trans_data/spm
CHECKPOINT_DIR=$SCRIPT_DIR/checkpoints/envi-transformer
BEST_CHECKPOINT=$CHECKPOINT_DIR/checkpoint_best.pt

OUT_DIR=output
SPLIT="test"  # You can change to valid, train, etc.

mkdir -p $OUT_DIR

# === Generate translations using the trained model ===
echo "Generating translations from the checkpoint..."

fairseq-generate $DATA_DIR \
  --path $BEST_CHECKPOINT \
  --batch-size 128 --beam 5 --remove-bpe \
  --source-lang $SRC --target-lang $TGT \
  --results-path $OUT_DIR \
  --gen-subset $SPLIT

echo "Translation generation complete! Output in: $OUT_DIR"
