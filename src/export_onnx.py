import argparse
import os
import torch
import sentencepiece as spm
from .model import Seq2SeqTransformer

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def export(checkpoint, sp_model, output_dir, d_model=512, nhead=8, num_layers=6, max_len=128):
    os.makedirs(output_dir, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    vocab_size = sp.vocab_size()
    model = Seq2SeqTransformer(vocab_size, d_model=d_model, nhead=nhead,
                              num_encoder_layers=num_layers, num_decoder_layers=num_layers).eval()

    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cpu")
    model.to(device)

    B, S, T = 1, max_len, max_len
    src = torch.randint(5, vocab_size, (B, S), device=device)
    tgt = torch.randint(5, vocab_size, (B, 1), device=device)  # start minimal T for tracing
    src_pad_mask = (src == 0)

    class EncWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, src, src_pad_mask):
            return self.m.encode(src, src_key_padding_mask=src_pad_mask)
    enc = EncWrapper(model).to(device).eval()
    enc_path = os.path.join(output_dir, "encoder.onnx")
    torch.onnx.export(enc, (src, src_pad_mask), enc_path,
                      input_names=["src", "src_pad_mask"],
                      output_names=["memory"],
                      dynamic_axes={"src": {0: "B", 1: "S"},
                                    "src_pad_mask": {0: "B", 1: "S"},
                                    "memory": {0: "B", 1: "S"}},
                      opset_version=17)
    print(f"Saved {enc_path}")

    class DecWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, tgt, memory, tgt_mask, tgt_pad_mask, mem_pad_mask):
            return self.m.decode(tgt, memory, tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_pad_mask,
                                memory_key_padding_mask=mem_pad_mask)
    dec = DecWrapper(model).to(device).eval()

    memory = torch.randn(B, S, d_model, device=device)
    tgt_pad_mask = (tgt == 0)
    mem_pad_mask = src_pad_mask
    # Provide a small (1x1) mask for tracing; declare T dynamic
    tgt_mask_trace = torch.zeros((1, 1), dtype=torch.float32, device=device)

    dec_path = os.path.join(output_dir, "decoder.onnx")
    torch.onnx.export(dec, (tgt, memory, tgt_mask_trace, tgt_pad_mask, mem_pad_mask), dec_path,
                      input_names=["tgt", "memory", "tgt_mask", "tgt_pad_mask", "mem_pad_mask"],
                      output_names=["logits"],
                      dynamic_axes={
                        "tgt": {0: "B", 1: "T"},
                        "memory": {0: "B", 1: "S"},
                        "tgt_mask": {0: "T", 1: "T"},
                        "tgt_pad_mask": {0: "B", 1: "T"},
                        "mem_pad_mask": {0: "B", 1: "S"},
                        "logits": {0: "B", 1: "T"}
                      },
                      opset_version=17)
    print(f"Saved {dec_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sp_model", required=True)
    ap.add_argument("--output_dir", default="onnx")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt.get("config", {})
    export(ckpt, args.sp_model, args.output_dir,
          d_model=cfg.get("d_model", args.d_model),
          nhead=cfg.get("nhead", args.nhead),
          num_layers=cfg.get("num_layers", args.num_layers),
          max_len=args.max_len)