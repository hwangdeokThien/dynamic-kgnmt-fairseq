from pathlib import Path
import argparse

ROOT_PATH = Path(__file__).resolve().parent.parent
EXTRACT_LOCAL_KG_PATH = ROOT_PATH / "extract-local-kg"
DYNAMIC_KGNMT_PATH = ROOT_PATH / "en-vi-dynamic-kgnmt"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default=None, help="train, valid, test")
    args = parser.parse_args()
    return args