import sentencepiece as spm
import os
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent

# Absolute paths
train_en = RUN_DIR / "kgnmt_data/train.en"
train_vi = RUN_DIR / "kgnmt_data/train.vi"
train_kg = RUN_DIR / "kgnmt_data/train.kg"
model_prefix = RUN_DIR / "tokenizer/envikg"

# Make sure the output directory exists
os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

special_tokens = [
    "<x>", "<t>", "<k>", "<r>",
    "<en>", "<vi>" 
]

# Train the tokenizer
spm.SentencePieceTrainer.Train(
    f'--input={train_en},{train_vi},{train_kg} '
    f'--model_prefix={model_prefix} '
    '--vocab_size=16000 '
    '--character_coverage=1.0 '
    '--model_type=bpe '
    f'--user_defined_symbols={",".join(special_tokens)} '
)
