import argparse, sacrebleu, sentencepiece as spm, torch
from tqdm import tqdm
from .model import Seq2SeqTransformer
from .export_onnx import load_checkpoint
from .data import load_dataset_by_name

def greedy_decode(model, sp, src_text, max_len=64, device="cpu"):
    model.eval()
    bos, eos, pad = 2, 3, 0
    src_ids = [bos] + sp.encode(src_text, out_type=int) + [eos]
    src_ids = src_ids[:max_len]
    src = torch.tensor([src_ids + [pad]*(max_len-len(src_ids))], device=device)
    src_pad_mask = (src == pad)
    memory = model.encode(src, src_pad_mask)
    ys = torch.tensor([[bos]], device=device)
    for _ in range(max_len-1):
        tgt_pad_mask = (ys == pad)
        T = ys.size(1)
        tgt_mask = torch.triu(torch.ones((T,T), device=device) * float("-inf"), diagonal=1)
        out = model.decode(ys, memory, tgt_mask, tgt_pad_mask, src_pad_mask)
        prob = out[:, -1, :].softmax(-1)
        next_tok = int(prob.argmax(-1).item())
        ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
        if next_tok == eos:
            break
    return sp.decode(list(ys[0].tolist()[1:-1]))

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = load_checkpoint(args.checkpoint)
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    vocab_size = sp.vocab_size()
    cfg = ckpt.get("config", {})
    model = Seq2SeqTransformer(vocab_size, d_model=cfg.get("d_model", 512),
                               nhead=cfg.get("nhead", 8),
                               num_encoder_layers=cfg.get("num_layers", 6),
                               num_decoder_layers=cfg.get("num_layers", 6)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = load_dataset_by_name(args.dataset, args.lang_pair)
    src, tgt = args.lang_pair.split("-")
    refs, hyps = [], []
    subset = ds["test"] if "test" in ds else (ds["validation"] if "validation" in ds else ds["train"])
    total = min(len(subset), args.n_samples)
    for i, ex in enumerate(tqdm(subset, total=total)):
        if i >= args.n_samples: break
        tr = ex.get("translation") or {}
        s, t = tr.get(src), tr.get(tgt)
        if not s or not t: continue
        hyp = greedy_decode(model, sp, s, max_len=args.max_len, device=device)
        hyps.append(hyp)
        refs.append(t)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"BLEU: {bleu.score:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sp_model", required=True)
    ap.add_argument("--dataset", default="opus_books")
    ap.add_argument("--lang_pair", default="en-de")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()
    main(args)