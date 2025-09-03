#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>

static const int PAD = 0;
static const int UNK = 1;
static const int BOS = 2;
static const int EOS = 3;

struct Args {
  std::string spm, encoder, decoder, src;
  int max_len = 64;
  int beam = 4;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1; i<argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i+1 >= argc) { std::cerr << "Missing value for " << k << "\n"; exit(1); }
      return argv[++i];
    };
    if (k == "--spm") a.spm = next();
    else if (k == "--encoder") a.encoder = next();
    else if (k == "--decoder") a.decoder = next();
    else if (k == "--src") a.src = next();
    else if (k == "--max_len") a.max_len = std::stoi(next());
    else if (k == "--beam") a.beam = std::stoi(next());
  }
  if (a.spm.empty() || a.encoder.empty() || a.decoder.empty() || a.src.empty()) {
    std::cerr << "Usage: mt_infer --spm sp.model --encoder encoder.onnx --decoder decoder.onnx --src \"text\" [--max_len 64 --beam 4]\n";
    exit(1);
  }
  return a;
}

std::vector<int64_t> pad_vec(const std::vector<int64_t>& v, int total, int pad=PAD) {
  std::vector<int64_t> out = v;
  out.resize(total, pad);
  return out;
}

int main(int argc, char** argv) {
  auto args = parse_args(argc, argv);

  sentencepiece::SentencePieceProcessor sp;
  auto status = sp.Load(args.spm);
  if (!status.ok()) {
    std::cerr << "Failed to load sentencepiece model: " << status.ToString() << "\n";
    return 1;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mt");
  Ort::SessionOptions so;
  OrtCUDAProviderOptions cuda_opts;
  so.AppendExecutionProvider_CUDA(cuda_opts);
  Ort::Session enc(env, args.encoder.c_str(), so);
  Ort::Session dec(env, args.decoder.c_str(), so);
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int> ids;
  sp.Encode(args.src, &ids);
  std::vector<int64_t> src_ids; src_ids.reserve(ids.size()+2);
  src_ids.push_back(BOS);
  for (int id : ids) src_ids.push_back((int64_t)id);
  src_ids.push_back(EOS);
  if ((int)src_ids.size() > args.max_len) src_ids.resize(args.max_len);
  auto src_padded = pad_vec(src_ids, args.max_len);

  std::vector<int64_t> src_pad_mask(args.max_len, false);
  for (int i = 0; i < args.max_len; ++i) if (src_padded[i] == PAD) src_pad_mask[i] = true;

  std::array<int64_t, 2> src_shape{1, args.max_len};
  std::array<int64_t, 2> mask_shape{1, args.max_len};

  Ort::Value src_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, src_padded.data(), src_padded.size(), src_shape.data(), src_shape.size());
  Ort::Value mask_tensor = Ort::Value::CreateTensor<bool>(mem_info, (bool*)src_pad_mask.data(), src_pad_mask.size(), mask_shape.data(), mask_shape.size());

  const char* enc_inputs[2] = {"src", "src_pad_mask"};
  const char* enc_outputs[1] = {"memory"};

  auto enc_out = enc.Run(Ort::RunOptions{nullptr}, enc_inputs, (const Ort::Value* const[]){src_tensor, mask_tensor}, 2, enc_outputs, 1);
  Ort::Value& memory = enc_out[0];

  struct Beam { double logp; std::vector<int64_t> seq; };
  std::vector<Beam> beams = {{0.0, {BOS}}};
  std::vector<Beam> completed;

  for (int step = 0; step < args.max_len-1; ++step) {
    std::vector<Beam> new_beams;
    for (auto& b : beams) {
      if (!b.seq.empty() && b.seq.back() == EOS) { completed.push_back(b); continue; }
      int T = (int)b.seq.size();
      auto tgt_padded = pad_vec(b.seq, args.max_len);

      std::vector<int64_t> tgt_pad_mask(args.max_len, false);
      for (int i=0;i<args.max_len;++i) if (tgt_padded[i]==PAD) tgt_pad_mask[i]=true;

      std::vector<float> tgt_mask(T*T, 0.f);
      for (int r=0;r<T;++r) for (int c=r+1;c<T;++c) tgt_mask[r*T + c] = -INFINITY;

      std::array<int64_t, 2> tgt_shape{1, args.max_len};
      std::array<int64_t, 2> tgt_pad_shape{1, args.max_len};
      std::array<int64_t, 2> tgt_mask_shape{T, T};

      Ort::Value tgt_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, tgt_padded.data(), tgt_padded.size(), tgt_shape.data(), tgt_shape.size());
      Ort::Value tgt_pad_tensor = Ort::Value::CreateTensor<bool>(mem_info, (bool*)tgt_pad_mask.data(), tgt_pad_mask.size(), tgt_pad_shape.data(), tgt_pad_shape.size());
      Ort::Value tgt_mask_tensor = Ort::Value::CreateTensor<float>(mem_info, tgt_mask.data(), tgt_mask.size(), tgt_mask_shape.data(), tgt_mask_shape.size());

      const char* dec_inputs[5] = {"tgt", "memory", "tgt_mask", "tgt_pad_mask", "mem_pad_mask"};
      const char* dec_outputs[1] = {"logits"};

      auto outs = dec.Run(Ort::RunOptions{nullptr}, dec_inputs,
                          (const Ort::Value* const[]){tgt_tensor, memory, tgt_mask_tensor, tgt_pad_tensor, mask_tensor},
                          5, dec_outputs, 1);
      auto& dec_out = outs[0];
      auto& ti = dec_out.GetTensorTypeAndShapeInfo();
      auto shape = ti.GetShape(); // (1,T,V)
      int64_t V = shape[2];

      float* logits = dec_out.GetTensorMutableData<float>();
      float* last = logits + (T-1)*V;

      std::vector<double> probs(V);
      double maxlog = -1e30;
      for (int64_t i=0;i<V;++i) maxlog = std::max(maxlog, (double)last[i]);
      double sum = 0.0;
      for (int64_t i=0;i<V;++i){ probs[i] = std::exp((double)last[i] - maxlog); sum += probs[i]; }
      for (int64_t i=0;i<V;++i) probs[i] /= (sum + 1e-12);

      std::vector<int64_t> idx(V);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin()+std::min(args.beam,(int)V), idx.end(),
                        [&](int64_t a, int64_t b){ return probs[a] > probs[b]; });
      int K = std::min(args.beam, (int)V);
      for (int k=0;k<K;++k){
        int64_t tok = idx[k];
        double lp = b.logp + std::log(probs[tok] + 1e-12);
        auto seq = b.seq; seq.push_back(tok);
        new_beams.push_back({lp, std::move(seq)});
      }
    }
    std::sort(new_beams.begin(), new_beams.end(), [](const Beam& a, const Beam& b){ return a.logp > b.logp; });
    if ((int)new_beams.size() > args.beam) new_beams.resize(args.beam);
    beams = new_beams;
    if ((int)completed.size() >= args.beam) break;
  }
  std::vector<Beam> final = completed;
  final.insert(final.end(), beams.begin(), beams.end());
  std::sort(final.begin(), final.end(), [](const Beam& a, const Beam& b){ return a.logp > b.logp; });
  auto best = final.front().seq;

  std::vector<int> out_ids;
  for (auto id : best) if (id != BOS && id != EOS) out_ids.push_back((int)id);
  std::string detok;
  sp.Decode(out_ids, &detok);
  std::cout << detok << std::endl;
  return 0;
}