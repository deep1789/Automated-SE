"""Merge Colab fine-tuning scores (colab/finetune_codebert.py) into the CodeBERT experiment cache.
    python src/merge_ft.py <colab_out_dir> [--cache results/cbn_cache]
Rows are aligned on the normalised-hash column `nh` (asserted), so a mismatch between the Colab universe and the local one fails loudly."""
import sys, glob, os, argparse
import pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("ft_dir"); ap.add_argument("--cache", default="results/cbn_cache"); a = ap.parse_args()
n = 0
for f in sorted(glob.glob(os.path.join(a.ft_dir, "*_s*_*.parquet"))):
    base = os.path.basename(f); tgt = os.path.join(a.cache, base)
    if not os.path.exists(tgt): print("no cache file for", base); continue
    ft, c = pd.read_parquet(f), pd.read_parquet(tgt)
    assert len(ft) == len(c) and (ft.nh.values == c.nh.values).all() and (ft.y.values == c.y.values).all(), f"row mismatch in {base}"
    c["ft_cb"] = ft.ft_cb.values; c.to_parquet(tgt); n += 1
print(f"merged {n} files into {a.cache}")
