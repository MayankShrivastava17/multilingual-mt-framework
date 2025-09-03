# Multilingual Machine Translation Framework (PyTorch + ONNX Runtime + FastAPI + C++)

 A compact, production‑style MT system that trains a Transformer on HF datasets, exports to ONNX, serves low‑latency inference via FastAPI, and ships a C++ client on ONNX Runtime, this was done as a project for Machine Learning class.

End-to-end machine translation capstone project: **train a Transformer model** on multilingual parallel corpora (e.g., OPUS Books or WMT), **export encoder/decoder to ONNX**, **serve via FastAPI** (GPU/CPU), and run **C++ inference** with ONNX Runtime.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

* **Default (auto‑downloaded by code):** [OPUS Books — Hugging Face](https://huggingface.co/datasets/opus_books)
* **Alternatives (also auto‑downloaded via HF):**

  * [WMT14](https://huggingface.co/datasets/wmt14) · [WMT16](https://huggingface.co/datasets/wmt16) · [WMT17](https://huggingface.co/datasets/wmt17)
* **Original sources (FYI):**

  * OPUS portal: [https://opus.nlpl.eu/](https://opus.nlpl.eu/)
  * WMT tasks: [https://www.statmt.org/wmt23/](https://www.statmt.org/wmt23/) (see “Shared Tasks” pages for older years)

> You select datasets via CLI flags (`--dataset` and `--lang_pair`). No manual download needed; HF Datasets handles caching.

---

## Training Pipeline


### 1) Train SentencePiece Tokenizer

```bash
set -euo pipefail
python -m src.tokenize \
  --dataset opus_books \
  --lang_pair en-de \
  --vocab_size 16000 \
  --output models/spm
```

### 2) Train Transformer Model

```bash
set -euo pipefail
python -m src.train \
  --dataset opus_books \
  --lang_pair en-de \
  --sp_model models/spm/en-de.model \
  --save_dir models/transformer_en-de \
  --epochs 5 \
  --batch_size 128 \
  --max_len 128
```

### 3) Export to ONNX

```bash
set -euo pipefail
python -m src.export_onnx \
  --checkpoint models/transformer_en-de/checkpoint.pt \
  --sp_model models/spm/en-de.model \
  --output_dir onnx \
  --max_len 128
```

---

## Run & Evaluate

> **Tip:** For CPU‑only boxes, install `onnxruntime` instead of `onnxruntime-gpu` — everything else works unchanged.

### Python Inference

```bash
python -m src.infer_python \
  --checkpoint models/transformer_en-de/checkpoint.pt \
  --sp_model models/spm/en-de.model \
  --src "Hello world!" --max_len 64
```

### BLEU Evaluation

```bash
python -m src.evaluate \
  --checkpoint models/transformer_en-de/checkpoint.pt \
  --sp_model models/spm/en-de.model \
  --dataset opus_books --lang_pair en-de \
  --n_samples 200 --max_len 64
```

---

## Serve FastAPI API

### Start server (GPU or CPU)

```bash
set -euo pipefail
cp -n .env.example .env || true
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Query API

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world!","lang_pair":"en-de","beam":4,"max_len":64}'
```

---

## C++ Inference Client

### Build

```bash
# Install sentencepiece dev libs (Debian/Ubuntu):
sudo apt-get install -y libsentencepiece-dev

# Download ONNX Runtime (GPU or CPU) that matches your CUDA (if GPU).
# Releases: https://github.com/microsoft/onnxruntime/releases
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-gpu-<ver>

cmake -S cpp -B build -DONNXRUNTIME_DIR=$ONNXRUNTIME_DIR
cmake --build build -j
```

### Run

```bash
./build/mt_infer \
  --spm models/spm/en-de.model \
  --encoder onnx/encoder.onnx \
  --decoder onnx/decoder.onnx \
  --src "Hello world!" --beam 4 --max_len 64
```

---

## All‑in‑one Bash 

If you want a **single** copy‑paste to do tokenizer → train → export → serve:

```bash
# === run_all.sh ===
set -euo pipefail

# 1) Tokenizer
python -m src.tokenize --dataset opus_books --lang_pair en-de --vocab_size 16000 --output models/spm

# 2) Train
python -m src.train --dataset opus_books --lang_pair en-de \
  --sp_model models/spm/en-de.model --save_dir models/transformer_en-de \
  --epochs 5 --batch_size 128 --max_len 128

# 3) Export ONNX
python -m src.export_onnx --checkpoint models/transformer_en-de/checkpoint.pt \
  --sp_model models/spm/en-de.model --output_dir onnx --max_len 128

# 4) Serve API
cp -n .env.example .env || true
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---


* **Dataset switch:** Use `--dataset wmt14 --lang_pair en-de` (or `wmt16`, `wmt17`).
* **Special tokens:** `pad=0, unk=1, bos=2, eos=3` are set in SentencePiece training and used by the model.
* **ONNX export:** dynamic axes for `B`, `S`, `T` allow variable lengths at inference.
* **CPU‑only:** swap `onnxruntime-gpu` → `onnxruntime` in `requirements.txt` and you’re good.
* **HF cache:** datasets are cached under `~/.cache/huggingface/datasets` by default.
