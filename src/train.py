import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from tqdm import tqdm
from .data import load_dataset_by_name
from .model import Seq2SeqTransformer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MTDataset(Dataset):
    def __init__(self, pairs, sp: spm.SentencePieceProcessor, max_len=128):
        self.pairs = pairs
        self.sp = sp
        self.max_len = max_len
        self.pad_id = 0
        self.bos = 2
        self.eos = 3

    def __len__(self):
        return len(self.pairs)

    def encode_line(self, text):
        ids = self.sp.encode(text, out_type=int)
        ids = [self.bos] + ids + [self.eos]
        return ids[: self.max_len]

    def pad(self, ids):
        return ids + [self.pad_id] * (self.max_len - len(ids))

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.encode_line(src)
        tgt_ids = self.encode_line(tgt)
        src_ids = self.pad(src_ids)
        tgt_in = self.pad(tgt_ids[:-1])  # decoder input (without EOS)
        tgt_out = self.pad(tgt_ids[1:])  # shifted target (without BOS)
        return torch.tensor(src_ids), torch.tensor(tgt_in), torch.tensor(tgt_out)

def create_dataloaders(dataset_name, lang_pair, sp_model, batch_size=128, max_len=128):
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    ds = load_dataset_by_name(dataset_name, lang_pair)
    src, tgt = lang_pair.split("-")

    def collect(split):
        data = []
        for ex in ds[split]:
            tr = ex.get("translation") or {}
            s = tr.get(src)
            t = tr.get(tgt)
            if s and t:
                data.append((s, t))
        return data

    train_pairs = collect("train")
    val_split = "validation" if "validation" in ds else ("test" if "test" in ds else "train")
    val_pairs = collect(val_split)[:5000]
    train_ds = MTDataset(train_pairs, sp, max_len)
    val_ds = MTDataset(val_pairs, sp, max_len)
    return sp, DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

def generate_square_subsequent_mask(sz: int, device):
    mask = torch.triu(torch.ones((sz, sz), device=device) * float("-inf"), diagonal=1)
    return mask

def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp, train_dl, val_dl = create_dataloaders(args.dataset, args.lang_pair, args.sp_model,
                                              batch_size=args.batch_size, max_len=args.max_len)
    vocab_size = sp.vocab_size()
    model = Seq2SeqTransformer(
        num_tokens=vocab_size, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
        dim_feedforward=2048, dropout=0.1
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    step = 0
    accum = args.accum_steps

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for src, tgt_in, tgt_out in tqdm(train_dl, desc=f"Epoch {epoch}"):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), device)
            logits = model(src, tgt_in, src_key_padding_mask=(src==0), tgt_mask=tgt_mask,
                           tgt_key_padding_mask=(tgt_in==0), memory_key_padding_mask=(src==0))
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            (loss/accum).backward()
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            running += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            count = 0
            for src, tgt_in, tgt_out in val_dl:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), device)
                logits = model(src, tgt_in, src_key_padding_mask=(src==0), tgt_mask=tgt_mask,
                               tgt_key_padding_mask=(tgt_in==0), memory_key_padding_mask=(src==0))
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
                val_loss += loss.item()
                count += 1
            val_loss /= max(count,1)
        print(f"Epoch {epoch}: train_loss={running/len(train_dl):.4f} val_loss={val_loss:.4f}")
        torch.save({"model_state": model.state_dict(),
                    "sp_model": args.sp_model,
                    "vocab_size": vocab_size,
                    "config": vars(args)}, os.path.join(args.save_dir, "checkpoint.pt"))
        if val_loss < best_val:
            best_val = val_loss

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="opus_books")
    ap.add_argument("--lang_pair", default="en-de")
    ap.add_argument("--sp_model", required=True)
    ap.add_argument("--save_dir", default="models/transformer_en-de")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--accum_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    train(args)