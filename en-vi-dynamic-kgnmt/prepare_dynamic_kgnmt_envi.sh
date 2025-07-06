#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "===== PREPARING EN-VI DATA ====="

# === Config ===
SRC=en
TGT=vi
KG=kg
DATA_DIR=$SCRIPT_DIR/kgnmt_data 
SPM_DATA_DIR=$DATA_DIR/spm
TOKENIZER_MODEL=$SCRIPT_DIR/tokenizer/envikg.model
OUT_DIR=$SCRIPT_DIR/data-bin/envi-bpe
SPLITS="train valid test"

mkdir -p "$OUT_DIR"
mkdir -p "$SPM_DATA_DIR"

# === Encode with SentencePiece ===
echo "Encoding with SentencePiece model: $TOKENIZER_MODEL"

for SPLIT in $SPLITS; do
    for LANG in $SRC $TGT $KG; do
        INPUT=$DATA_DIR/$SPLIT.$LANG
        OUTPUT=$SPM_DATA_DIR/$SPLIT.spm.$LANG

        echo "Encoding $INPUT → $OUTPUT"

        python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('$TOKENIZER_MODEL')
with open('$INPUT', 'r', encoding='utf-8') as fin, open('$OUTPUT', 'w', encoding='utf-8') as fout:
    for line in fin:
        fout.write(' '.join(sp.encode(line.strip(), out_type=str)) + '\n')
"
    done
done

# === Binarize with fairseq-preprocess ===
echo "Running fairseq-preprocess..."

fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $SPM_DATA_DIR/train.spm \
  --validpref $SPM_DATA_DIR/valid.spm \
  --testpref  $SPM_DATA_DIR/test.spm \
  --destdir $OUT_DIR \
  --joined-dictionary \
  --workers 20

fairseq-preprocess \
  --source-lang $KG --target-lang $KG \
  --trainpref $SPM_DATA_DIR/train.spm \
  --validpref $SPM_DATA_DIR/valid.spm \
  --testpref  $SPM_DATA_DIR/test.spm \
  --destdir $OUT_DIR \
  --joined-dictionary \
  --workers 20

mv $OUT_DIR/train.kg-kg.kg.bin $OUT_DIR/train.en-vi.kg.bin
mv $OUT_DIR/valid.kg-kg.kg.bin $OUT_DIR/valid.en-vi.kg.bin
mv $OUT_DIR/test.kg-kg.kg.bin $OUT_DIR/test.en-vi.kg.bin
mv $OUT_DIR/train.kg-kg.kg.idx $OUT_DIR/train.en-vi.kg.idx
mv $OUT_DIR/valid.kg-kg.kg.idx $OUT_DIR/valid.en-vi.kg.idx
mv $OUT_DIR/test.kg-kg.kg.idx $OUT_DIR/test.en-vi.kg.idx

echo "✅ Data preparation complete: $OUT_DIR"