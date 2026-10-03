"""Deterministic reconstruction of the CodeBERT-experiment universe and scenario splits (shared by the CPU pipeline and the Colab fine-tuning).
Needs only data/clean.parquet (from prep.py). No embeddings or hashed features required."""
import os, re, json, hashlib
import numpy as np, pandas as pd
from run_models import SC, make_split

NPOS, NNEG = 5000, 10000
DATASETS = ("DiverseVul", "BigVul")
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
def normtxt(c): return re.sub(r"\s+", " ", COMMENT.sub(" ", c)).strip()

def write_meta(clean="data/clean.parquet"):
    """data/meta_{ds}.parquet exactly as feats.py writes it (clean.parquet rows of that dataset, without the code column)."""
    df = pd.read_parquet(clean, columns=["ds", "y", "project", "commit_id", "cwe", "year", "nh"])
    for ds in DATASETS:
        p = f"data/meta_{ds}.parquet"
        if not os.path.exists(p): df[df.ds == ds].reset_index(drop=True).to_parquet(p)

def universe(ds):
    meta = pd.read_parquet(f"data/meta_{ds}.parquet")
    rng = np.random.RandomState(123)
    pos = np.where(meta.y.values == 1)[0]; neg = np.where(meta.y.values == 0)[0]
    sp = np.sort(rng.choice(pos, min(NPOS, len(pos)), replace=False)); sn = np.sort(rng.choice(neg, NNEG, replace=False))
    idx = np.concatenate([sp, sn])
    u = meta.iloc[idx].copy(); u["src_idx"] = idx
    u["w"] = np.where(u.y.values == 1, len(pos) / len(sp), len(neg) / len(sn))
    return u.reset_index(drop=True)

def universe_texts(ds, u, clean="data/clean.parquet"):
    codes = pd.read_parquet(clean, columns=["ds", "code"]); codes = codes[codes.ds == ds].code.values[u.src_idx.values]
    return [normtxt(c) for c in codes]

def parts_for(name, seed, mS, mT):
    S, T, kind = SC[name]
    parts = make_split(name, seed, mS, mT)
    if kind == "cross":   # drop any target function that occurs anywhere in the full source corpus
        full = set(pd.read_parquet(f"data/meta_{S}.parquet", columns=["nh"]).nh.values)
        for k in ("cal_tgt", "test"):
            if k in parts:
                w_, idx = parts[k]; parts[k] = (w_, idx[~mT.nh.iloc[idx].isin(full).values])
        parts.pop("test_leaky", None)
    return parts

def checksum(u): return hashlib.md5("".join(u.nh.values).encode()).hexdigest()
