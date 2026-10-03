"""Formatting-only probe on PrimeVul (raw text) - does the Big-Vul shortcut exist there?"""
import json, re, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
def fmt(c):
    n = max(len(c), 1); lines = c.split("\n"); nl = max(len(lines), 1)
    return [c.count("\t")/n, c.count("\r")/n, sum(l.endswith((" ", "\t")) for l in lines)/nl, c.count("/*")+c.count("//"), len(c), nl,
            float(c.rstrip() != c), float(c.startswith((" ", "\t", "\n"))), float(c.endswith("\n")), c.count("  ")/n, sum(l.startswith("    ") for l in lines)/nl,
            sum(l.startswith("\t") for l in lines)/nl, float("{" in lines[0]) if lines else 0, np.mean([len(l) for l in lines]), float(bool(re.search(r"\n\{\s*\n", c)))]
def main():
    rows = []
    for f in ("train", "valid", "test"):
        for l in open(f"data/raw/PrimeVul/primevul_{f}.jsonl"):
            d = json.loads(l); rows.append((d["func"], int(d["target"])))
    df = pd.DataFrame(rows, columns=["code", "y"]); rng = np.random.RandomState(0)
    pos = df[df.y == 1]; neg = df[df.y == 0].sample(30000, random_state=0); s = pd.concat([pos, neg]); w = np.where(s.y == 1, len(pos)/len(pos), (df.y == 0).sum()/30000)
    F = np.array([fmt(c) for c in s.code.values], dtype=np.float32); y = s.y.values
    tr, te = train_test_split(np.arange(len(y)), test_size=.3, random_state=0, stratify=y)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1).fit(F[tr], y[tr]); p = m.predict_proba(F[te])[:, 1]
    out = dict(auroc=roc_auc_score(y[te], p), auprc_weighted=average_precision_score(y[te], p, sample_weight=w[te]))
    for nm, i in (("trailing_ws_lines", 2), ("comment_markers", 3), ("starts_with_ws", 7), ("brace_own_line", 14)): out[nm] = dict(vulnerable=float(F[y == 1, i].mean()), benign=float(F[y == 0, i].mean()))
    json.dump(out, open("results/format_audit_pv.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
