"""Hashed lexical features (stateless) + engineered code metrics."""
import re, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer

TOKEN = r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|&&|\|\||->|\+\+|--|<<|>>|[^\s\w]"
NFEAT = 2 ** 19
HV = HashingVectorizer(n_features=NFEAT, ngram_range=(1, 2), token_pattern=TOKEN, lowercase=False,
                       alternate_sign=False, norm=None, dtype=np.float32)
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
DANGER = ["strcpy", "strncpy", "strcat", "strncat", "sprintf", "snprintf", "vsprintf", "gets", "scanf", "sscanf",
          "memcpy", "memmove", "memset", "alloca", "malloc", "calloc", "realloc", "free", "read", "recv", "fread",
          "system", "popen", "exec", "atoi", "strlen", "sizeof", "printf", "fprintf", "open", "fopen", "assert"]
KW = ["if", "else", "for", "while", "do", "switch", "case", "goto", "return", "break", "continue"]

def strip(code): return COMMENT.sub(" ", code)

def metrics(code: str) -> list:
    c = strip(code)
    toks = re.findall(TOKEN, c)
    cnt = {}
    for t in toks: cnt[t] = cnt.get(t, 0) + 1
    depth = mx = 0
    for ch in c:
        if ch == "{": depth += 1; mx = max(mx, depth)
        elif ch == "}": depth -= 1
    lines = c.count("\n") + 1
    f = [len(c), lines, len(toks), mx]
    f += [cnt.get(k, 0) for k in KW]
    f += [cnt.get(k, 0) for k in DANGER]
    f += [cnt.get(k, 0) for k in ["*", "&", "->", "[", "(", "+", "-", "<", ">", "=", "==", "!=", "&&", "||", "?", "++", "--", "<<", ">>"]]
    f += [1 + sum(cnt.get(k, 0) for k in ["if", "for", "while", "case", "&&", "||", "?"])]
    f += [len(re.findall(r"\(\s*(?:unsigned\s+|const\s+|struct\s+)?\w+\s*\*+\s*\)", c)), len(re.findall(r'"[^"\n]*"', c)),
          len(re.findall(r"\b\d+\b", c)), len(set(t for t in toks if re.match(r"[A-Za-z_]", t)))]
    return f
METRIC_NAMES = (["chars", "lines", "tokens", "max_nesting"] + [f"kw_{k}" for k in KW] + [f"api_{k}" for k in DANGER] +
                [f"op_{k}" for k in ["*", "&", "->", "[", "(", "+", "-", "<", ">", "=", "==", "!=", "&&", "||", "?", "++", "--", "<<", ">>"]] +
                ["cyclomatic", "ptr_casts", "string_lits", "num_lits", "uniq_idents"])

def _chunk(codes):
    X = HV.transform([strip(c) for c in codes]).tocsr()
    M = np.array([metrics(c) for c in codes], dtype=np.float32)
    return X, M

def featurize(codes, n_jobs=4, chunk=5000):
    parts = Parallel(n_jobs=n_jobs)(delayed(_chunk)(codes[i:i + chunk]) for i in range(0, len(codes), chunk))
    return sp.vstack([p[0] for p in parts]).tocsr(), np.vstack([p[1] for p in parts])

if __name__ == "__main__":
    df = pd.read_parquet("data/clean.parquet")
    for ds in ["DiverseVul", "BigVul"]:
        sub = df[df.ds == ds].reset_index(drop=True)
        X, M = featurize(sub["code"].tolist())
        sp.save_npz(f"data/X_{ds}.npz", X); np.save(f"data/M_{ds}.npy", M)
        sub.drop(columns=["code"]).to_parquet(f"data/meta_{ds}.parquet")
        print(ds, X.shape, X.nnz, flush=True)
