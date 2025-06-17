# thesis - Thien and Truyen

#!/bin/bash

set -e

# ==== Step 0: Config ====
SRC=en
TGT=vi
VOCAB_SIZE=8000
MODEL_PREFIX=envi
DATA_DIR=trans_data
BIN_DIR=data-bin/envi-bpe
SP_MODEL=$MODEL_PREFIX.model
NUM_WORKERS=4

# ==== Step 1: Train SentencePiece ====
echo "Training SentencePiece model..."
spm_train --input=$DATA_DIR/train.$SRC,$DATA_DIR/train.$TGT \
          --model_prefix=$MODEL_PREFIX \
          --vocab_size=$VOCAB_SIZE \
          --character_coverage=1.0 \
          --model_type=bpe

# ==== Step 2: Encode all splits ====
for SPLIT in train valid test; do
    for LANG in $SRC $TGT; do
        echo "Encoding $SPLIT.$LANG..."
        spm_encode --model=$SP_MODEL --output_format=piece \
                   < $DATA_DIR/$SPLIT.$LANG > $DATA_DIR/$SPLIT.spm.$LANG
    done
done

# ==== Step 3: Preprocess ====
echo "Binarizing data..."
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $DATA_DIR/train.spm \
  --validpref $DATA_DIR/valid.spm \
  --testpref  $DATA_DIR/test.spm \
  --destdir $BIN_DIR \
  --joined-dictionary \
  --workers $NUM_WORKERS

# ==== Step 4: Train Transformer ====
echo "Training Transformer model..."
CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
  data-bin/envi-bpe \
  --distributed-world-size 2 \
  --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.3 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4096 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --save-dir checkpoints/envi-transformer

echo "Training complete!"
