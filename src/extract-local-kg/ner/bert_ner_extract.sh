# thesis - Thien and Truyen

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
EXTRACT_KG_DIR="$PROJECT_DIR/extract-local-kg"
DYNAMIC_KGNMT_DIR="$PROJECT_DIR/dynamic-kgnmt-fairseq"

SPLITS="train valid test"

for SPLIT in $SPLITS; do
    python kg_BERT_NER.py --split $SPLIT
    python preprocess_kg_NER.py --split $SPLIT
    python extract_id_entity.py --split $SPLIT
    python extract_triple_id.py --split $SPLIT
    python decode_triple.py --split $SPLIT
    echo "✅ Processed $SPLIT"
    cp $EXTRACT_KG_DIR/data-kg/$SPLIT/extracted_triples.txt $DYNAMIC_KGNMT_DIR/en-vi-dynamic-kgnmt/kgnmt_data/$SPLIT.kg
done
