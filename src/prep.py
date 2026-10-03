"""Load DiverseVul + Big-Vul, normalise, deduplicate, audit leakage, write clean parquet."""
import glob, hashlib, json, re
import pandas as pd

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
def norm(code: str) -> str:
    return re.sub(r"\s+", " ", COMMENT.sub(" ", code)).strip()
def h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()

def load_dv():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/raw/DiverseVul/data/*.parquet"))])
    df = df.rename(columns={"func": "code", "target": "y"})
    df["ds"] = "DiverseVul"; df["year"] = pd.NA
    df["cwe"] = df["cwe"].astype(str)
    return df[["ds", "code", "y", "project", "commit_id", "cwe", "year"]]

def load_bv():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/raw/bigvul/data/*.parquet"))])
    df = df.rename(columns={"func_before": "code", "vul": "y", "CWE ID": "cwe"})
    df["year"] = df["CVE ID"].str.extract(r"CVE-(\d{4})-")[0].astype("Int64")
    df["ds"] = "BigVul"
    df["y"] = df["y"].astype(int)
    return df[["ds", "code", "y", "project", "commit_id", "cwe", "year"]]

def dedup(df):
    df = df.copy()
    df["nh"] = [h(norm(c)) for c in df["code"]]
    n0 = len(df)
    g = df.groupby("nh")["y"].agg(["nunique", "size"])
    conflict = g.index[g["nunique"] > 1]
    n_conf_rows = int(df["nh"].isin(conflict).sum())
    exact_dup_rows = int(n0 - df["nh"].nunique())
    df = df[~df["nh"].isin(conflict)].drop_duplicates("nh").reset_index(drop=True)
    return df, dict(raw=n0, exact_dup_rows_removed=exact_dup_rows,
                    label_conflict_hashes=int(len(conflict)), label_conflict_rows=n_conf_rows,
                    clean=len(df), vuln=int(df.y.sum()), vuln_rate=float(df.y.mean()),
                    projects=int(df.project.nunique()))

if __name__ == "__main__":
    audit = {}
    out = []
    for name, loader in [("DiverseVul", load_dv), ("BigVul", load_bv)]:
        raw = loader()
        raw = raw[raw["code"].str.len().between(20, 20000)]
        audit[name] = {"rows_after_len_filter": len(raw)}
        clean, st = dedup(raw)
        audit[name].update(st)
        out.append(clean)
    dv, bv = out
    shared = set(dv.nh) & set(bv.nh)
    audit["cross_dataset_shared_functions"] = len(shared)
    audit["shared_frac_of_bigvul"] = len(shared) / len(bv)
    audit["shared_frac_of_diversevul"] = len(shared) / len(dv)
    audit["shared_vuln_in_bigvul"] = int(bv[bv.nh.isin(shared)].y.sum())
    audit["shared_vuln_in_diversevul"] = int(dv[dv.nh.isin(shared)].y.sum())
    allc = pd.concat(out, ignore_index=True)
    allc.to_parquet("data/clean.parquet")
    json.dump(audit, open("results/audit.json", "w"), indent=2)
    print(json.dumps(audit, indent=2))
    print(bv.year.value_counts().sort_index().to_string())
