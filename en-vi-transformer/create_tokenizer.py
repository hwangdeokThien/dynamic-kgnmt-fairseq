import sentencepiece as spm
import os
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent

# Absolute paths
train_en = RUN_DIR / "trans_data/train.en"
train_vi = RUN_DIR / "trans_data/train.vi"
model_prefix = RUN_DIR / "tokenizer/envi"

# Make sure the output directory exists
os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

# Train the tokenizer
spm.SentencePieceTrainer.Train(
    f'--input={train_en},{train_vi} '
    f'--model_prefix={model_prefix} '
    '--vocab_size=8000 '
    '--character_coverage=1.0 '
    '--model_type=bpe'
)
