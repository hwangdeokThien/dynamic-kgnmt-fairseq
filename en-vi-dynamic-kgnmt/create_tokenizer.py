import sentencepiece as spm
import os
from pathlib import Path

# --- Config paths ---
RUN_DIR = Path(__file__).resolve().parent
train_en = RUN_DIR / "kgnmt_data/train.en"
train_vi = RUN_DIR / "kgnmt_data/train.vi"
train_kg = RUN_DIR / "kgnmt_data/train.kg"
model_prefix = RUN_DIR / "tokenizer/envikg"
output_dict_path = RUN_DIR / "tokenizer/dict.envikg.txt"

# Ensure tokenizer output directory exists
os.makedirs(model_prefix.parent, exist_ok=True)

# Special tokens to preserve and inject
special_tokens = ["<x>", "<t>", "<k>", "<r>", "<en>", "<vi>"]

# --- Train SentencePiece tokenizer ---
spm.SentencePieceTrainer.Train(
    f'--input={train_en},{train_vi},{train_kg} '
    f'--model_prefix={model_prefix} '
    '--vocab_size=8000 '
    '--character_coverage=1.0 '
    '--model_type=bpe '
    f'--user_defined_symbols={",".join(special_tokens)} '
)

print("✅ SentencePiece tokenizer trained.")

# --- Convert SentencePiece vocab to Fairseq dict format ---
spm_vocab_path = f"{model_prefix}.vocab"

with open(spm_vocab_path, 'r', encoding='utf-8') as fin, open(output_dict_path, 'w', encoding='utf-8') as fout:
    # Add special tokens first (required by Fairseq)
    fout.write('<pad> 1 #fairseq:overwrite\n')
    fout.write('<unk> 1 #fairseq:overwrite\n')
    fout.write('<s> 1 #fairseq:overwrite\n')
    fout.write('</s> 1 #fairseq:overwrite\n')

    added = {'<pad>', '<unk>', '<s>', '</s>'}
    for line in fin:
        token, _ = line.strip().split('\t')
        if token in added:
            continue  # Skip duplicates
        fout.write(f'{token} 1\n')
        added.add(token)

print(f"✅ Fairseq dictionary saved to {output_dict_path}")
