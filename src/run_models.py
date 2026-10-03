"""Train detectors for one (scenario, seed); write scores for cal/test parts to results/cache."""
import sys, time, os, json
import numpy as np, pandas as pd, scipy.sparse as sp
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

SC = {  # name -> (source, target, split kind)
    "DV-R": ("DiverseVul", "DiverseVul", "random"), "DV-P": ("DiverseVul", "DiverseVul", "project"),
    "BV-R": ("BigVul", "BigVul", "random"), "BV-P": ("BigVul", "BigVul", "project"),
    "BV-T": ("BigVul", "BigVul", "temporal"),
    "DV2BV": ("DiverseVul", "BigVul", "cross"), "BV2DV": ("BigVul", "DiverseVul", "cross"),
    # third corpus (PrimeVul)
    "PV-R": ("PrimeVul", "PrimeVul", "random"), "PV-P": ("PrimeVul", "PrimeVul", "project"), "PV-T": ("PrimeVul", "PrimeVul", "temporal"),
    "DV2PV": ("DiverseVul", "PrimeVul", "cross"), "BV2PV": ("BigVul", "PrimeVul", "cross"), "PV2DV": ("PrimeVul", "DiverseVul", "cross"), "PV2BV": ("PrimeVul", "BigVul", "cross"),
}
TEMPORAL = {"BigVul": (2015, 2016, 2017), "PrimeVul": (2018, 2019, 2020)}   # train <= a; target pool == b; test >= c (by CVE year)
MAXNEG = 150_000

def load(ds):
    X = sp.load_npz(f"data/X_{ds}.npz"); M = np.load(f"data/M_{ds}.npy"); meta = pd.read_parquet(f"data/meta_{ds}.parquet")
    return X, M, meta

def make_split(name, seed, metaS, metaT):
    rng = np.random.RandomState(seed)
    S, T, kind = SC[name]
    parts = {}
    if kind == "random":
        r = rng.rand(len(metaS))
        parts = dict(train=("S", np.where(r < .70)[0]), cal_id=("S", np.where((r >= .70) & (r < .85))[0]), test=("S", np.where(r >= .85)[0]))
    elif kind == "project":
        pr = metaS.project.unique(); rng.shuffle(pr); k = int(.7 * len(pr))
        trp, tep = set(pr[:k]), set(pr[k:])
        tr = np.where(metaS.project.isin(trp))[0]; te = np.where(metaS.project.isin(tep))[0]
        r = rng.rand(len(tr)); cal = tr[r < .15]; tr = tr[r >= .15]
        r2 = rng.rand(len(te)); pool = te[r2 < .25]; te = te[r2 >= .25]
        parts = dict(train=("S", tr), cal_id=("S", cal), cal_tgt=("S", pool), test=("S", te))
    elif kind == "temporal":
        yr = metaS.year.astype(float).values
        a_, b_, c_ = TEMPORAL[SC[name][0]]
        tr = np.where(yr <= a_)[0]; r = rng.rand(len(tr)); cal = tr[r < .15]; tr = tr[r >= .15]
        parts = dict(train=("S", tr), cal_id=("S", cal), cal_tgt=("S", np.where(yr == b_)[0]), test=("S", np.where(yr >= c_)[0]))
    elif kind == "cross":
        r = rng.rand(len(metaS)); tr = np.where(r < .85)[0]; cal = np.where(r >= .85)[0]
        trh = set(metaS.nh.values[tr])
        leak = metaT.nh.isin(trh).values
        r2 = rng.rand(len(metaT)); pool_m = (r2 < .25) & ~leak; test_m = (r2 >= .25) & ~leak
        parts = dict(train=("S", tr), cal_id=("S", cal), cal_tgt=("T", np.where(pool_m)[0]), test=("T", np.where(test_m)[0]),
                     test_leaky=("T", np.where((r2 >= .25) & leak)[0]))
    return parts

def cap_train(idx, y, seed):
    rng = np.random.RandomState(seed + 99)
    pos = idx[y[idx] == 1]; neg = idx[y[idx] == 0]
    if len(neg) > MAXNEG: neg = rng.choice(neg, MAXNEG, replace=False)
    out = np.concatenate([pos, neg]); rng.shuffle(out); return out

def domain_weights(Ztr_src, Ztr_tgt, seed):
    """Cross-fitted density-ratio estimate w(x)=p_t(x)/p_s(x) via logistic domain classifier."""
    ns, nt = Ztr_src.shape[0], Ztr_tgt.shape[0]
    Z = sp.vstack([Ztr_src, Ztr_tgt]).tocsr(); d = np.r_[np.zeros(ns), np.ones(nt)]
    pr = np.zeros(len(d)); skf = StratifiedKFold(2, shuffle=True, random_state=seed)
    for a, b in skf.split(Z, d):
        clf = LogisticRegression(C=1.0, max_iter=200, solver="liblinear").fit(Z[a], d[a]); pr[b] = clf.predict_proba(Z[b])[:, 1]
    pr = np.clip(pr, 1e-3, 1 - 1e-3)
    w = pr / (1 - pr) * (ns / nt)
    return np.clip(w, 0.05, 20.0)[:ns], np.clip(w, 0.05, 20.0)[ns:]

def main(name, seed, models):
    t0 = time.time()
    S, T, kind = SC[name]
    XS, MS, mS = load(S); XT, MT, mT = (XS, MS, mS) if T == S else load(T)
    parts = make_split(name, seed, mS, mT)
    getX = lambda w: (XS, MS, mS) if w == "S" else (XT, MT, mT)
    tr_idx = cap_train(parts["train"][1], mS.y.values, seed)
    Xtr, Mtr, ytr = XS[tr_idx], MS[tr_idx], mS.y.values[tr_idx]
    tf = TfidfTransformer(sublinear_tf=True).fit(Xtr)
    Ttr = tf.transform(Xtr)
    out = {}
    def feats_for(which, idx):
        X, M, m = getX(which); return tf.transform(X[idx]), X[idx], M[idx], m.iloc[idx]
    eval_parts = {k: v for k, v in parts.items() if k != "train"}
    cache = {k: feats_for(*v) for k, v in eval_parts.items()}
    scores = {k: {} for k in eval_parts}
    if "lr" in models:
        lr = LogisticRegression(C=20, solver="liblinear", max_iter=200).fit(Ttr, ytr)
        for k in cache: scores[k]["lr"] = lr.decision_function(cache[k][0])
    if "svm" in models:
        sv = LinearSVC(C=0.3, max_iter=2000).fit(Ttr, ytr)
        for k in cache: scores[k]["svm"] = sv.decision_function(cache[k][0])
    if "lgbm" in models:
        df_ = np.asarray((Xtr > 0).sum(0)).ravel(); keep = np.argsort(-df_)[:20000]
        mk = lambda X, M: sp.hstack([sp.csr_matrix(M), X[:, keep]]).tocsr().astype(np.float32)
        rng = np.random.RandomState(seed); r = rng.rand(len(ytr)) < .9
        A = mk(Xtr, Mtr)
        mdl = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.08, num_leaves=63, colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                 min_child_samples=20, reg_lambda=1.0, n_jobs=int(os.environ.get("NJ", 4)), random_state=seed, verbose=-1)
        mdl.fit(A[r], ytr[r], eval_set=[(A[~r], ytr[~r])], eval_metric="auc", callbacks=[lgb.early_stopping(40, verbose=False)])
        for k in cache: scores[k]["lgbm"] = mdl.predict_proba(mk(cache[k][1], cache[k][2]))[:, 1]
        mdl.booster_.save_model(f"results/cache/{name}_s{seed}_lgbm.txt")
        out["lgbm_iters"] = int(mdl.best_iteration_ or 600)
    # domain-shift weights (cal_id vs test), only when the target differs from the source distribution
    if kind != "random":
        ws, wt = domain_weights(cache["cal_id"][0], cache["test"][0], seed)
        scores["cal_id"]["w"] = ws; scores["test"]["w"] = wt
        if "cal_tgt" in cache: pass
    for k in cache:
        d = cache[k][3][["ds", "y", "project", "cwe", "year", "nh"]].copy().reset_index(drop=True)
        d["length"] = cache[k][2][:, 0]
        for mname, v in scores[k].items(): d[mname] = v
        d["part"] = k; d["scenario"] = name; d["seed"] = seed
        d.to_parquet(f"results/cache/{name}_s{seed}_{k}.parquet")
    out.update(time=time.time() - t0, train_n=len(tr_idx), train_pos=int(ytr.sum()), **{f"n_{k}": int(len(v[1])) for k, v in eval_parts.items()})
    json.dump(out, open(f"results/cache/{name}_s{seed}_info.json", "w"))
    print(name, seed, out, flush=True)

if __name__ == "__main__":
    os.makedirs("results/cache", exist_ok=True)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","))
