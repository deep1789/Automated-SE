"""Flawfinder (rule-based static analyser) baseline on a random test subsample per scenario."""
import os, csv, io, subprocess, sys, shutil, glob
import numpy as np, pandas as pd
from joblib import Parallel, delayed

SCR = os.environ.get("SCR", "/tmp/flaw")
def run_dir(d):
    p = subprocess.run(["flawfinder", "--csv", "--quiet", "--dataonly", "-m", "0", d], capture_output=True, text=True)
    hits = {}
    for row in csv.DictReader(io.StringIO(p.stdout)):
        k = os.path.basename(row["File"]); lvl = int(row["Level"])
        mx, n = hits.get(k, (0, 0)); hits[k] = (max(mx, lvl), n + 1)
    return hits

def main(sc, seed=0, nsamp=25000):
    te = pd.read_parquet(f"results/cache/{sc}_s{seed}_test.parquet")
    te = te.sample(min(nsamp, len(te)), random_state=1).reset_index(drop=True)
    codes = pd.read_parquet("data/clean.parquet", columns=["nh", "code"]).drop_duplicates("nh").set_index("nh").code
    shutil.rmtree(f"{SCR}/{sc}", ignore_errors=True)
    dirs = []
    for i, nh in enumerate(te.nh):
        d = f"{SCR}/{sc}/{i // 2500}"; os.makedirs(d, exist_ok=True)
        if d not in dirs: dirs.append(d)
        open(f"{d}/{i}.c", "w").write(codes[nh] + "\n")
    res = Parallel(n_jobs=int(os.environ.get("FJ", 4)))(delayed(run_dir)(d) for d in dirs)
    allh = {}
    for r in res: allh.update(r)
    lv = np.array([allh.get(f"{i}.c", (0, 0))[0] for i in range(len(te))]); nh_ = np.array([allh.get(f"{i}.c", (0, 0))[1] for i in range(len(te))])
    te["flaw_level"] = lv; te["flaw_hits"] = nh_
    te["flaw"] = lv * 100 + nh_
    te.to_parquet(f"results/flaw_{sc}.parquet")
    print(sc, len(te), "pos", te.y.sum(), "frac with hit", (lv > 0).mean(), flush=True)

if __name__ == "__main__":
    for sc in sys.argv[1:]: main(sc)
