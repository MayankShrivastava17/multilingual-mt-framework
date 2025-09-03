import argparse
from datasets import load_dataset


def load_opus_books(lang_pair: str):
  assert "-" in lang_pair, "lang_pair like 'en-de'"
  src_lang, tgt_lang = lang_pair.split("-")
  # opus_books uses translation field with nested lang keys
  ds = load_dataset("opus_books", f"{src_lang}-{tgt_lang}")
  return ds


def load_dataset_by_name(name: str, lang_pair: str):
  if name == "opus_books":
    return load_opus_books(lang_pair)
  elif name in {"wmt14", "wmt16", "wmt17"}:
    src_lang, tgt_lang = lang_pair.split("-")
    config = f"{src_lang}-{tgt_lang}"
    return load_dataset(name, config)
  else:
    raise ValueError(f"Unsupported dataset: {name}")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", default="opus_books")
  ap.add_argument("--lang_pair", default="en-de")
  args = ap.parse_args()
  ds = load_dataset_by_name(args.dataset, args.lang_pair)
  print(ds)