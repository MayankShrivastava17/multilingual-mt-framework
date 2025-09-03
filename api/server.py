from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import sentencepiece as spm
import os
import torch

app = FastAPI(title="MT Server (ONNXRuntime GPU)")

class Req(BaseModel):
    text: str
    lang_pair: str = "en-de"
    max_len: int = 64
    beam: int = 4
    length_penalty: float = 1.0  # optional quality knob

SPM_PATH = os.getenv("SPM_PATH", "models/spm/en-de.model")
ENC_PATH = os.getenv("ENC_PATH", "onnx/encoder.onnx")
DEC_PATH = os.getenv("DEC_PATH", "onnx/decoder.onnx")

sp = spm.SentencePieceProcessor(model_file=SPM_PATH)
providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
enc_sess = ort.InferenceSession(ENC_PATH, providers=providers)
dec_sess = ort.InferenceSession(DEC_PATH, providers=providers)

BOS, EOS, PAD = 2, 3, 0

def beam_search(src_text, max_len=64, beam=4, length_penalty=1.0):
    src_ids = [BOS] + sp.encode(src_text, out_type=int) + [EOS]
    if len(src_ids) > max_len:
        src_ids = src_ids[:max_len]
    src = np.array([src_ids + [PAD]*(max_len - len(src_ids))], dtype=np.int64)
    src_pad = (src == PAD)

    memory = enc_sess.run(["memory"], {"src": src, "src_pad_mask": src_pad})[0]

    beams = [(0.0, [BOS])]
    completed = []

    for _ in range(max_len-1):
        new_beams = []
        for lp, seq in beams:
            if seq[-1] == EOS:
                completed.append((lp, seq))
                continue
            T = len(seq)
            tgt = np.array([seq + [PAD]*(max_len-T)], dtype=np.int64)
            tgt_pad = (tgt == PAD)
            tgt_mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)

            logits = dec_sess.run(["logits"], {
                "tgt": tgt[:, :T],
                "memory": memory,
                "tgt_mask": tgt_mask,
                "tgt_pad_mask": tgt_pad[:, :T],
                "mem_pad_mask": src_pad
            })[0]
            last = logits[0, T-1]
            probs = torch.softmax(torch.from_numpy(last), dim=-1).numpy()
            topk = np.argpartition(-probs, beam)[:beam]
            for tok in topk:
                new_lp = lp + float(np.log(probs[tok] + 1e-9))
                new_seq = seq + [int(tok)]
                new_beams.append((new_lp, new_seq))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam]
        if len(completed) >= beam:
            break

    final = completed + beams
    # length normalization (optional)
    final = [ (lp / (len(seq) ** length_penalty if length_penalty > 0 else 1.0), seq) for lp, seq in final ]
    final.sort(key=lambda x: x[0], reverse=True)
    best = final[0][1]
    ids = [i for i in best if i not in (BOS, EOS)]
    return sp.decode(ids)

@app.post("/translate")
def translate(req: Req):
    text = req.text.strip()
    out = beam_search(text, max_len=req.max_len, beam=req.beam, length_penalty=req.length_penalty)
    return {"translation": out}