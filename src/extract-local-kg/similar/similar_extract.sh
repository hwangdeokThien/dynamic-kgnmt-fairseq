# thesis - Thien and Truyen

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
EXTRACT_KG_DIR="$PROJECT_DIR/extract-local-kg"
DYNAMIC_KGNMT_DIR="$PROJECT_DIR/dynamic-kgnmt-fairseq"

SPLITS="train valid test"

for SPLIT in $SPLITS; do
    python decode_all_triple.py --split $SPLIT
    python langdetect.py --split $SPLIT
    python decode_langdetect_triple.py --split $SPLIT
    python extract_popular_rare_random.py --split $SPLIT
    python kg_similar.py --split $SPLIT
    echo "✅ Processed $SPLIT"
    echo "src: $EXTRACT_KG_DIR/data-kg/$SPLIT/all.kg"
    echo "tgt: $DYNAMIC_KGNMT_DIR/en-vi-dynamic-kgnmt/kgnmt_data/$SPLIT.kg"
    cp $EXTRACT_KG_DIR/data-kg/$SPLIT/all.kg $DYNAMIC_KGNMT_DIR/en-vi-dynamic-kgnmt/kgnmt_data/$SPLIT.kg
done
