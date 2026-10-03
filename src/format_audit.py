"""Formatting-shortcut audit: how well do raw-text formatting features alone predict the label? + effect of size/format features in the main LightGBM."""
import json, re, numpy as np, pandas as pd, scipy.sparse as sp, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from run_models import load, make_split, cap_train
NAMES = ["tabs_per_char", "cr_per_char", "trailing_ws_lines", "comment_markers", "chars", "lines", "trailing_ws_end", "starts_with_ws", "ends_newline",
         "double_spaces", "indent4_lines", "indent_tab_lines", "brace_on_first_line", "mean_line_len", "brace_own_line"]
out = {}
for ds in ("BigVul", "DiverseVul"):
    F = np.load(f"data/cb/fmt_{ds}.npy"); u = pd.read_parquet(f"data/cb/meta_{ds}.parquet"); y = u.y.values; w = u.w.values
    tr, te = train_test_split(np.arange(len(y)), test_size=.3, random_state=0, stratify=y)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1).fit(F[tr], y[tr]); p = m.predict_proba(F[te])[:, 1]
    d = dict(auroc=roc_auc_score(y[te], p), auprc_weighted=average_precision_score(y[te], p, sample_weight=w[te]))
    for nm, i in (("trailing_ws_lines", 2), ("comment_markers", 3), ("starts_with_ws", 7), ("brace_own_line", 14)):
        d[nm] = dict(vulnerable=float(F[y == 1, i].mean()), benign=float(F[y == 0, i].mean()))
    out[ds] = d
# does the main LightGBM depend on size/format-sensitive metrics?
chk = {}
for ds, sc in (("BigVul", "BV-R"), ("DiverseVul", "DV-R")):
    X, M, meta = load(ds); seed = 0; parts = make_split(sc, seed, meta, meta); tr = cap_train(parts["train"][1], meta.y.values, seed); te = parts["test"][1]
    keep = np.argsort(-np.asarray((X[tr] > 0).sum(0)).ravel())[:20000]; res = {}
    for label, cols in (("all", []), ("drop_chars_lines", [0, 1]), ("drop_chars_lines_tokens", [0, 1, 2])):
        M2 = M.copy(); M2[:, cols] = 0
        mk = lambda Xs, Ms: sp.hstack([sp.csr_matrix(Ms), Xs[:, keep]]).tocsr().astype(np.float32)
        r = np.random.RandomState(seed).rand(len(tr)) < .9
        g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, n_jobs=4, random_state=seed, verbose=-1)
        g.fit(mk(X[tr], M2[tr])[r], meta.y.values[tr][r]); p = g.predict_proba(mk(X[te], M2[te]))[:, 1]
        res[label] = dict(auroc=roc_auc_score(meta.y.values[te], p), auprc=average_precision_score(meta.y.values[te], p))
    chk[ds] = res
out["main_lgbm_check"] = chk
json.dump(out, open("results/format_audit.json", "w"), indent=1); print(json.dumps(out, indent=1)[:1500])
