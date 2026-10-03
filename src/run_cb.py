"""Detectors on frozen CodeBERT embeddings (+ matched TF-IDF baselines) on the stratified subsample, same scenarios/splits as run_models."""
import sys, os, time, json
import numpy as np, pandas as pd, scipy.sparse as sp
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from run_models import SC, make_split

EMB = os.environ.get("EMB", "raw"); PFX = "emb" if EMB == "raw" else "embn"; CACHE = "results/cb_cache" if EMB == "raw" else "results/cbn_cache"
def load_u(ds):
    meta = pd.read_parquet(f"data/cb/meta_{ds}.parquet").reset_index(drop=True)
    E = np.load(f"data/cb/{PFX}_{ds}.npy"); assert np.load(f"data/cb/{PFX}done_{ds}.npy").all()
    F = np.load(f"data/cb/fmt_{ds}.npy")
    X = sp.load_npz(f"data/X_{ds}.npz")[meta.src_idx.values]; M = np.load(f"data/M_{ds}.npy")[meta.src_idx.values]
    return meta, E, X, M, F

def main(name, seed):
    t0 = time.time(); S, T, kind = SC[name]
    mS, ES, XS, MS, FS = load_u(S); mT, ET, XT, MT, FT = (mS, ES, XS, MS, FS) if T == S else load_u(T)
    parts = make_split(name, seed, mS, mT)
    if kind == "cross":   # stricter than the main study: drop any target function that occurs anywhere in the full source corpus
        full = set(pd.read_parquet(f"data/meta_{S}.parquet", columns=["nh"]).nh.values)
        for k in ("cal_tgt", "test", "test_leaky"):
            if k in parts:
                w_, idx = parts[k]; keep = ~mT.nh.iloc[idx].isin(full).values
                parts[k] = (w_, idx[keep]) if k != "test_leaky" else (w_, idx[~keep])
    tr = parts["train"][1]; ytr = mS.y.values[tr]
    sc = StandardScaler().fit(ES[tr]); Ztr = sc.transform(ES[tr])
    cb_lr = LogisticRegression(C=0.05, max_iter=1000).fit(Ztr, ytr)
    rng = np.random.RandomState(seed); r = rng.rand(len(tr)) < .9
    A = np.hstack([ES[tr], MS[tr]]).astype(np.float32)
    cb_gb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, n_jobs=int(os.environ.get("NJ", 4)), random_state=seed, verbose=-1)
    cb_gb.fit(A[r], ytr[r], eval_set=[(A[~r], ytr[~r])], eval_metric="auc", callbacks=[lgb.early_stopping(40, verbose=False)])
    tf = TfidfTransformer(sublinear_tf=True).fit(XS[tr]); tf_lr = LogisticRegression(C=20, solver="liblinear", max_iter=200).fit(tf.transform(XS[tr]), ytr)
    dfq = np.asarray((XS[tr] > 0).sum(0)).ravel(); keep = np.argsort(-dfq)[:20000]
    mk = lambda X, M: sp.hstack([sp.csr_matrix(M), X[:, keep]]).tocsr().astype(np.float32)
    tf_gb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, n_jobs=int(os.environ.get("NJ", 4)), random_state=seed, verbose=-1)
    B = mk(XS[tr], MS[tr]); tf_gb.fit(B[r], ytr[r], eval_set=[(B[~r], ytr[~r])], eval_metric="auc", callbacks=[lgb.early_stopping(40, verbose=False)])
    fm_gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, n_jobs=int(os.environ.get("NJ", 4)), random_state=seed, verbose=-1).fit(FS[tr], ytr)
    info = {}
    for k, (which, idx) in parts.items():
        if k == "train": continue
        E_, X_, M_, m_, F_ = (ES, XS, MS, mS, FS) if which == "S" else (ET, XT, MT, mT, FT)
        d = m_.iloc[idx][["y", "w", "project", "year", "nh"]].copy().reset_index(drop=True)
        d["length"] = M_[idx][:, 0]
        d["cb_lr"] = cb_lr.decision_function(sc.transform(E_[idx]))
        d["cb_gb"] = cb_gb.predict_proba(np.hstack([E_[idx], M_[idx]]).astype(np.float32))[:, 1]
        d["tf_lr"] = tf_lr.decision_function(tf.transform(X_[idx]))
        d["tf_gb"] = tf_gb.predict_proba(mk(X_[idx], M_[idx]))[:, 1]
        d["fmt_gb"] = fm_gb.predict_proba(F_[idx])[:, 1]
        d["part"] = k; d["scenario"] = name; d["seed"] = seed
        d.to_parquet(f"{CACHE}/{name}_s{seed}_{k}.parquet"); info[f"n_{k}"] = int(len(idx)); info[f"pos_{k}"] = int(d.y.sum())
    info["time"] = time.time() - t0; info["train_n"] = int(len(tr)); info["train_pos"] = int(ytr.sum())
    json.dump(info, open(f"{CACHE}/{name}_s{seed}_info.json", "w")); print(name, seed, info, flush=True)

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True); main(sys.argv[1], int(sys.argv[2]))
