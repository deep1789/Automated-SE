"""Which part of the 'formatting' signal is whitespace structure vs size vs comment density? (three corpora, raw text)"""
import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from format_audit_pv import fmt
GROUPS = {"size (chars, lines, mean line length)": [4, 5, 13], "comment markers": [3], "whitespace structure": [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 14]}
def auc(F, y, cols):
    tr, te = train_test_split(np.arange(len(y)), test_size=.3, random_state=0, stratify=y)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1).fit(F[tr][:, cols], y[tr]); return float(roc_auc_score(y[te], m.predict_proba(F[te][:, cols])[:, 1]))
data = {}
for ds in ("BigVul", "DiverseVul"):
    data[ds] = (np.load(f"data/cb/fmt_{ds}.npy"), pd.read_parquet(f"data/cb/meta_{ds}.parquet").y.values)
rows = []
for f in ("train", "valid", "test"):
    for l in open(f"data/raw/PrimeVul/primevul_{f}.jsonl"):
        d = json.loads(l); rows.append((d["func"], int(d["target"])))
df = pd.DataFrame(rows, columns=["code", "y"]); s = pd.concat([df[df.y == 1], df[df.y == 0].sample(30000, random_state=0)])
data["PrimeVul"] = (np.array([fmt(c) for c in s.code.values], dtype=np.float32), s.y.values)
out = {}
for ds, (F, y) in data.items():
    allc = list(range(F.shape[1])); r = {"all": auc(F, y, allc)}
    for g, cols in GROUPS.items():
        r[f"only {g}"] = auc(F, y, cols); r[f"all except {g}"] = auc(F, y, [c for c in allc if c not in cols])
    out[ds] = r; print(ds, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open("results/format_ablation.json", "w"), indent=1)
