"""Online risk control of the clearance rule under drift: static, sliding-window recalibration, adaptive conformal inference (ACI), and ACI+SW.
Stream = test functions in arrival order, processed in batches; labels (vulnerable or not) of a batch arrive after the batch (delayed feedback)."""
import sys, glob
import numpy as np, pandas as pd
from triage import threshold_for_alpha
sys.path.insert(0, "src")
from analyze import add_ens

def thr(cal_pos, a):
    if a <= 0: return -np.inf
    if a >= 1: return np.inf
    return threshold_for_alpha(cal_pos, a)

def run_stream(s, y, cal_pos, alpha, method, batch=1000, gamma=0.02, W=200, kmin=50):
    """Returns per-function cleared flags (bool) and the alpha_t trajectory per batch."""
    n = len(s); cleared = np.zeros(n, bool); a_t = alpha; buf = []; traj = []
    for st in range(0, n, batch):
        sl = slice(st, min(st + batch, n)); sb, yb = s[sl], y[sl]
        use_sw = method in ("sw", "aci_sw") and len(buf) >= kmin
        ref = np.array(buf[-W:]) if use_sw else cal_pos
        level = a_t if method in ("aci", "aci_sw") else alpha
        t = thr(ref, level); cb = sb <= t; cleared[sl] = cb; traj.append((a_t, t))
        # delayed feedback: labels of this batch become known after it was processed
        pos = yb == 1
        if method in ("aci", "aci_sw"):
            for e in cb[pos]: a_t = a_t + gamma * (alpha - float(e))
        if method in ("sw", "aci_sw"): buf.extend(sb[pos].tolist())
    return cleared, traj

def summarise(cleared, y, alpha, win=300):
    pos = y == 1; e = cleared[pos].astype(float)
    roll = pd.Series(e).rolling(win).mean().dropna().values if len(e) >= win else np.array([e.mean()])
    return dict(miss=e.mean(), sar=cleared.mean(), worst_win=roll.max(), frac_win_over=float((roll > alpha + 0.05).mean()), n_pos=int(pos.sum()), n=len(y))

def order_for(sc, te, seed):
    rng = np.random.RandomState(seed)
    if sc == "BV-T":
        m = pd.read_parquet("data/bv_cve_order.parquet")[["nh", "year", "num"]]; j = te.merge(m, on="nh", how="left", suffixes=("", "_r"))
        return np.lexsort((j.num.fillna(0).values, j.year.values))
    if sc in ("BV-P", "DV-P", "PV-P"):   # projects arrive one after another
        pr = te.project.unique().copy(); rng.shuffle(pr); rank = {p: i for i, p in enumerate(pr)}
        return np.lexsort((rng.rand(len(te)), te.project.map(rank).values))
    return rng.permutation(len(te))   # exchangeable / one-time shift

if __name__ == "__main__":
    rows, traj_save = [], {}
    gammas = [0.005, 0.01, 0.02, 0.05]
    for sc in ["BV-T", "BV-P", "DV-P", "DV2BV", "BV2DV", "DV-R", "BV-R"]:
        for seed in range(3):
            fs = glob.glob(f"results/cache/{sc}_s{seed}_*.parquet")
            P = {f.split(f"_s{seed}_")[1][:-8]: pd.read_parquet(f) for f in fs}; P = add_ens(P)
            cal, te = P["cal_id"], P["test"]; o = order_for(sc, te, seed); s = te.ens.values[o]; y = te.y.values[o]; cal_pos = cal.ens.values[cal.y.values == 1]
            for alpha in (0.05, 0.10):
                variants = [("static", None), ("sw", None), ("aci_sw", 0.02)] + [("aci", g) for g in gammas]
                for method, g in variants:
                    cl, traj = run_stream(s, y, cal_pos, alpha, method, gamma=g or 0.02)
                    r = summarise(cl, y, alpha); r.update(scenario=sc, seed=seed, alpha=alpha, method=method, gamma=g)
                    if method == "aci":
                        b = max(int((y[i:i + 1000] == 1).sum()) for i in range(0, len(y), 1000)); r["bound"] = (1 + g * b) / (g * r["n_pos"])
                    rows.append(r)
                    if sc == "BV-T" and seed == 0 and alpha == 0.10 and method in ("static", "sw", "aci", "aci_sw") and (g in (None, 0.02)):
                        pos_e = cl[y == 1].astype(float); traj_save[method] = pd.Series(pos_e).rolling(300).mean().values.tolist()
            print(sc, seed, flush=True)
    pd.DataFrame(rows).to_csv("results/online.csv", index=False)
    import json; json.dump(traj_save, open("results/online_traj.json", "w"))
