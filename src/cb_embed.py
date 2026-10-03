"""CodeBERT embeddings (frozen) for a stratified subsample of both corpora. Resumable; CPU, length-sorted batches."""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
torch.set_num_threads(4)
NPOS, NNEG, MAXLEN, BS = 5000, 10000, 256, 32
MODE = sys.argv[1] if len(sys.argv) > 1 else "raw"   # raw text or comment/whitespace-normalised text
PFX = "emb" if MODE == "raw" else "embn"
import re
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
normtxt = lambda c: re.sub(r"\s+", " ", COMMENT.sub(" ", c)).strip()
os.makedirs("data/cb", exist_ok=True)

def universe():
    out = []
    for ds in ("DiverseVul", "BigVul"):
        meta = pd.read_parquet(f"data/meta_{ds}.parquet")
        rng = np.random.RandomState(123)
        pos = np.where(meta.y.values == 1)[0]; neg = np.where(meta.y.values == 0)[0]
        sp = np.sort(rng.choice(pos, min(NPOS, len(pos)), replace=False)); sn = np.sort(rng.choice(neg, NNEG, replace=False))
        idx = np.concatenate([sp, sn])
        u = meta.iloc[idx].copy(); u["src_idx"] = idx
        u["w"] = np.where(u.y.values == 1, len(pos) / len(sp), len(neg) / len(sn))
        out.append(u.reset_index(drop=True))
    return out

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base"); model = AutoModel.from_pretrained("microsoft/codebert-base").eval()
    codes_all = pd.read_parquet("data/clean.parquet", columns=["ds", "code"])
    for u, ds in zip(universe(), ("DiverseVul", "BigVul")):
        u.drop(columns=[]).to_parquet(f"data/cb/meta_{ds}.parquet")
        codes = codes_all[codes_all.ds == ds].code.values[u.src_idx.values]
        if MODE != "raw": codes = [normtxt(c) for c in codes]
        enc = tok([c for c in codes], truncation=True, max_length=MAXLEN)["input_ids"]
        lens = np.array([len(x) for x in enc]); order = np.argsort(lens)
        path = f"data/cb/{PFX}_{ds}.npy"; done_path = f"data/cb/{PFX}done_{ds}.npy"
        E = np.load(path) if os.path.exists(path) else np.zeros((len(u), 1536), np.float32)
        done = np.load(done_path) if os.path.exists(done_path) else np.zeros(len(u), bool)
        t0 = time.time(); n0 = int(done.sum())
        for bi, i in enumerate(range(0, len(order), BS)):
            idx = order[i:i + BS]
            if done[idx].all(): continue
            b = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt")
            with torch.inference_mode():
                h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            mean = (h * m).sum(1) / m.sum(1)
            E[idx] = torch.cat([h[:, 0], mean], 1).numpy(); done[idx] = True
            if bi % 20 == 0:
                np.save(path, E); np.save(done_path, done)
                print(ds, int(done.sum()), "/", len(u), "%.1f seq/s" % ((done.sum() - n0) / (time.time() - t0)), flush=True)
        np.save(path, E); np.save(done_path, done); print(ds, "finished", flush=True)
