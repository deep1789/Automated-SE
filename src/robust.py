"""Decision consistency of the triage layer under (approximately) semantics-preserving transformations."""
import re, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from run_models import load, make_split, cap_train
from feats import _chunk, strip
from triage import threshold_for_alpha

KW = set("""auto break case char const continue default do double else enum extern float for goto if inline int long register return
short signed sizeof static struct switch typedef union unsigned void volatile while class public private protected new delete
this template typename namespace using bool true false NULL nullptr throw try catch""".split())
DECL = re.compile(r"\b((?:const|unsigned|signed|static|struct|enum|volatile|register|long|short)\s+)*([A-Za-z_]\w*)(?:\s*\*+\s*|\s+)([A-Za-z_]\w*)\s*(?=[=;,\[)])")

def rename_locals(code):
    c = strip(code)
    names = []
    for m in DECL.finditer(c):
        t, n = m.group(2), m.group(3)
        if t in KW and t not in ("int", "char", "long", "short", "unsigned", "signed", "float", "double", "void", "bool", "struct", "enum", "const", "static") : continue
        if n in KW or n.isupper() or n in names: continue
        names.append(n)
    for i, n in enumerate(names):
        c = re.sub(r"(?<![\w.])(?<!->)\b%s\b(?!\s*\()" % re.escape(n), f"v{i}_", c)
    return c

def dead_code(code):
    c = strip(code); i = c.find("{")
    return c if i < 0 else c[:i + 1] + "\n\tif (0) { int unused_0 = 0; unused_0++; }\n" + c[i + 1:]

TRANS = {"original": lambda c: c, "rename": rename_locals, "deadcode": dead_code, "both": lambda c: dead_code(rename_locals(c))}

def main(sc, seed=0, alpha=0.10):
    S = "DiverseVul" if sc == "DV-R" else "BigVul"
    XS, MS, mS = load(S)
    parts = make_split(sc, seed, mS, mS)
    tr = cap_train(parts["train"][1], mS.y.values, seed)
    tf = TfidfTransformer(sublinear_tf=True).fit(XS[tr]); y = mS.y.values
    lr = LogisticRegression(C=20, solver="liblinear", max_iter=200).fit(tf.transform(XS[tr]), y[tr])
    df_ = np.asarray((XS[tr] > 0).sum(0)).ravel(); keep = np.argsort(-df_)[:20000]
    mk = lambda X, M: sp.hstack([sp.csr_matrix(M), X[:, keep]]).tocsr().astype(np.float32)
    rng = np.random.RandomState(seed); r = rng.rand(len(tr)) < .9
    A = mk(XS[tr], MS[tr])
    gb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=63, colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                            n_jobs=4, random_state=seed, verbose=-1).fit(A[r], y[tr][r])
    score = {"lr": lambda X, M: lr.decision_function(tf.transform(X)), "lgbm": lambda X, M: gb.predict_proba(mk(X, M))[:, 1]}
    ci = parts["cal_id"][1]
    thr = {k: threshold_for_alpha(f(XS[ci], MS[ci])[y[ci] == 1], alpha) for k, f in score.items()}
    te = parts["test"][1]; pos = te[y[te] == 1]; neg = te[y[te] == 0]
    sel = np.concatenate([rng.choice(pos, min(len(pos), 1200), replace=False), rng.choice(neg, 4000, replace=False)])
    codes = pd.read_parquet("data/clean.parquet", columns=["nh", "ds", "code"]); codes = codes[codes.ds == S].reset_index(drop=True)
    assert (codes.nh.values == mS.nh.values).all()
    ys = y[sel]; rows = []; base = {}
    for tname, fn in TRANS.items():
        X2, M2 = _chunk([fn(codes.code.values[i]) for i in sel])
        for k, f in score.items():
            s = f(X2, M2)
            if tname == "original": base[k] = s
            passed = s <= thr[k]; passed0 = base[k] <= thr[k]
            rows.append(dict(scenario=sc, model=k, transform=tname, auroc=roc_auc_score(ys, s), miss=float(passed[ys == 1].mean()),
                             clear_neg=float(passed[ys == 0].mean()), flip=float((passed != passed0).mean()),
                             flip_pos=float((passed != passed0)[ys == 1].mean()),
                             rank_corr=float(pd.Series(s).corr(pd.Series(base[k]), method="spearman"))))
        print(sc, tname, flush=True)
    pd.DataFrame(rows).to_csv(f"results/robust_{sc}.csv", index=False)

if __name__ == "__main__":
    for sc in sys.argv[1:]: main(sc)
