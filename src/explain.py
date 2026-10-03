"""Explanation fidelity / stability (metrics-only LightGBM + TreeSHAP) and token-level shortcut analysis (LR)."""
import numpy as np, pandas as pd, scipy.sparse as sp, json
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from run_models import load, make_split, cap_train
from feats import METRIC_NAMES, HV, TOKEN
import re

def fit_m(M, y, seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=4, random_state=seed, verbose=-1).fit(M, y)

def shap_abs(model, M):
    c = model.booster_.predict(M, pred_contrib=True)
    return c[:, :-1]

def main():
    out, imp, fid_rows, perf = {}, {}, [], []
    data = {ds: load(ds) for ds in ("DiverseVul", "BigVul")}
    Ms = {}
    for ds, sc in (("DiverseVul", "DV-R"), ("BigVul", "BV-R")):
        X, M, m = data[ds]
        for seed in range(3):
            parts = make_split(sc, seed, m, m); tr = cap_train(parts["train"][1], m.y.values, seed); te = parts["test"][1]
            mdl = fit_m(M[tr], m.y.values[tr], seed)
            p = mdl.predict_proba(M[te])[:, 1]; yt = m.y.values[te]
            perf.append(dict(ds=ds, seed=seed, auroc=roc_auc_score(yt, p), auprc=average_precision_score(yt, p)))
            # cross-dataset (leakage-removed) transfer of the interpretable model
            other = "BigVul" if ds == "DiverseVul" else "DiverseVul"; Xo, Mo, mo = data[other]
            keep = ~mo.nh.isin(set(m.nh.values[tr])).values
            po = mdl.predict_proba(Mo[keep])[:, 1]; yo = mo.y.values[keep]
            perf.append(dict(ds=f"{ds}->{other}", seed=seed, auroc=roc_auc_score(yo, po), auprc=average_precision_score(yo, po)))
            rng = np.random.RandomState(seed); si = rng.choice(len(te), 4000, replace=False)
            S = shap_abs(mdl, M[te][si]); imp[(ds, seed)] = np.abs(S).mean(0)
            if seed == 0:
                Ms[ds] = (mdl, M[te][si], S, M[tr], yt[si])
                # deletion fidelity: neutralise top-k attributed features (replace by train median) on predicted-vulnerable fns
                med = np.median(M[tr], 0); base = mdl.booster_.predict(M[te][si], raw_score=True)
                top = np.quantile(base, 0.9); sel = np.where(base >= top)[0]
                gimp = mdl.booster_.feature_importance("gain"); gorder = np.argsort(-gimp)
                for k in range(0, 11):
                    r_shap, r_rand, r_gain = [], [], []
                    for i in sel:
                        x = M[te][si][i].copy(); xs = x.copy(); xr = x.copy(); xg = x.copy()
                        o = np.argsort(-np.abs(S[i]))[:k]; xs[o] = med[o]
                        xr[rng.choice(len(x), k, replace=False)] = med[rng.choice(len(x), k, replace=False)] if False else med[rng.choice(len(x), k, replace=False)]
                        xg[gorder[:k]] = med[gorder[:k]]
                        r_shap.append(xs); r_rand.append(xr); r_gain.append(xg)
                    for name, arr in (("TreeSHAP (local)", r_shap), ("Random", r_rand), ("Global gain", r_gain)):
                        d = base[sel] - mdl.booster_.predict(np.array(arr), raw_score=True)
                        fid_rows.append(dict(ds=ds, k=k, method=name, drop=float(d.mean())))
    pd.DataFrame(perf).to_csv("results/explain_perf.csv", index=False)
    pd.DataFrame(fid_rows).to_csv("results/explain_fidelity.csv", index=False)
    # stability of global importance rankings
    rows = []
    for ds in ("DiverseVul", "BigVul"):
        for a in range(3):
            for b in range(a + 1, 3): rows.append(dict(kind="within-dataset (seed pairs)", ds=ds, rho=spearmanr(imp[(ds, a)], imp[(ds, b)])[0]))
    for a in range(3):
        for b in range(3): rows.append(dict(kind="cross-dataset (DV vs BV)", ds="DV-BV", rho=spearmanr(imp[("DiverseVul", a)], imp[("BigVul", b)])[0]))
    pd.DataFrame(rows).to_csv("results/explain_stability.csv", index=False)
    mean_imp = pd.DataFrame({"feature": METRIC_NAMES, "DiverseVul": np.mean([imp[("DiverseVul", s)] for s in range(3)], 0),
                             "BigVul": np.mean([imp[("BigVul", s)] for s in range(3)], 0)})
    mean_imp.to_csv("results/explain_importance.csv", index=False)
    # shortcut analysis: are top lexical features project-specific?
    X, M, m = data["DiverseVul"]
    parts = make_split("DV-R", 0, m, m); tr = cap_train(parts["train"][1], m.y.values, 0)
    tf = TfidfTransformer(sublinear_tf=True).fit(X[tr]); lr = LogisticRegression(C=20, solver="liblinear", max_iter=200).fit(tf.transform(X[tr]), m.y.values[tr])
    coef = lr.coef_.ravel()
    Xb = (X[tr] > 0).tocsc(); proj = m.project.values[tr]
    dfreq = np.asarray(Xb.sum(0)).ravel()
    cand = np.where(dfreq >= 20)[0]
    top = cand[np.argsort(-coef[cand])[:200]]
    def nproj(j):
        rows = Xb.indices[Xb.indptr[j]:Xb.indptr[j + 1]]; return len(set(proj[rows]))
    tp = np.array([nproj(j) for j in top])
    rr = np.random.RandomState(0)
    # matched control: random features with the same document-frequency decile
    ctrl = []
    for j in top:
        pool = cand[np.abs(np.log(dfreq[cand]) - np.log(dfreq[j])) < 0.2]; ctrl.append(rr.choice(pool))
    cp = np.array([nproj(j) for j in ctrl])
    # token names for unigrams
    sample = m.index[:0]
    codes = pd.read_parquet("data/clean.parquet", columns=["ds", "code"]); codes = codes[codes.ds == "DiverseVul"].code.values
    vocab = {}
    for c in codes[tr[:40000]]:
        for t in set(re.findall(TOKEN, c)): vocab[t] = vocab.get(t, 0) + 1
    toks = [t for t, n in vocab.items() if n >= 5]
    idx = {}
    H = HV.transform(toks).tocsr()
    for t, i in zip(toks, H.indices[H.indptr[:-1]]): idx.setdefault(i, []).append((vocab[t], t))
    names = {i: max(v)[1] for i, v in idx.items()}
    toplist = [(names.get(j, "<bigram/other>"), float(coef[j]), int(dfreq[j]), int(nproj(j))) for j in np.argsort(-np.where(dfreq >= 20, coef, -9))[:25]]
    json.dump(dict(top_median_projects=float(np.median(tp)), ctrl_median_projects=float(np.median(cp)),
                   top_le5=float((tp <= 5).mean()), ctrl_le5=float((cp <= 5).mean()), top_tokens=toplist), open("results/explain_shortcut.json", "w"), indent=1)
    print(json.dumps(dict(top_med=float(np.median(tp)), ctrl_med=float(np.median(cp)), top_le5=float((tp <= 5).mean()), ctrl_le5=float((cp <= 5).mean()))))

if __name__ == "__main__":
    main()
