"""PrimeVul (third corpus): load jsonl, normalise, deduplicate, audit overlap with DiverseVul/Big-Vul -> data/clean_pv.parquet, results/audit_pv.json"""
import json, numpy as np, pandas as pd
from prep import norm, h, dedup
rows = []
for f in ("train", "valid", "test"):
    for l in open(f"data/raw/PrimeVul/primevul_{f}.jsonl"):
        d = json.loads(l)
        rows.append((d["func"], int(d["target"]), d["project"], d["commit_id"], str(d.get("cwe")), d["cve"]))
df = pd.DataFrame(rows, columns=["code", "y", "project", "commit_id", "cwe", "cve"])
df["year"] = df.cve.str.extract(r"CVE-(\d{4})-")[0].astype("Int64"); df["ds"] = "PrimeVul"
df = df[df["code"].str.len().between(20, 20000)]
n_len = len(df)
clean, st = dedup(df[["ds", "code", "y", "project", "commit_id", "cwe", "year"]])
st["rows_after_len_filter"] = n_len
base = pd.read_parquet("data/clean.parquet", columns=["ds", "nh", "y"])
for other in ("DiverseVul", "BigVul"):
    o = base[base.ds == other]; sh = set(o.nh) & set(clean.nh)
    st[f"shared_with_{other}"] = len(sh); st[f"shared_frac_of_PrimeVul_{other}"] = len(sh) / len(clean); st[f"shared_vuln_in_PrimeVul_{other}"] = int(clean[clean.nh.isin(sh)].y.sum())
clean.to_parquet("data/clean_pv.parquet"); json.dump(st, open("results/audit_pv.json", "w"), indent=2); print(json.dumps(st, indent=2))
print(clean.year.value_counts().sort_index().tail(12).to_string())
