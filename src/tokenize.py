import argparse
import sentencepiece as spm
from .data import load_dataset_by_name
import os

def train_sentencepiece(dataset_name: str, lang_pair: str, vocab_size: int, output_dir: str):
    ds = load_dataset_by_name(dataset_name, lang_pair)
    src_lang, tgt_lang = lang_pair.split("-")
    os.makedirs(output_dir, exist_ok=True)
    src_path = os.path.join(output_dir, f"{lang_pair}.src.txt")
    tgt_path = os.path.join(output_dir, f"{lang_pair}.tgt.txt")

    with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
        for split in ds:
            for ex in ds[split]:
                tr = ex.get("translation") or {}
                s = tr.get(src_lang)
                t = tr.get(tgt_lang)
                if s and t:
                    fs.write(s.replace("\n"," ") + "\n")
                    ft.write(t.replace("\n"," ") + "\n")

    combined = os.path.join(output_dir, f"{lang_pair}.all.txt")
    with open(combined, "w", encoding="utf-8") as f:
        with open(src_path, "r", encoding="utf-8") as fs:
            f.write(fs.read())
        with open(tgt_path, "r", encoding="utf-8") as ft:
            f.write(ft.read())

    model_prefix = os.path.join(output_dir, lang_pair)
    spm.SentencePieceTrainer.Train(
        input=combined,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=[]
    )
    print(f"Saved SentencePiece model to {model_prefix}.model")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="opus_books")
    ap.add_argument("--lang_pair", default="en-de")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--output", default="models/spm")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    train_sentencepiece(args.dataset, args.lang_pair, args.vocab_size, args.output)