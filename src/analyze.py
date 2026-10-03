"""Compute all result tables (CSV) from cached scores."""
import glob, json, ast, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from triage import *
from scipy.stats import ks_2samp

ALPHAS = [0.02, 0.05, 0.10, 0.20]
KS = [25, 50, 100, 200, 400]
R = 200
NCAL = 300
rng = np.random.RandomState(7)

import os
SCEN_ENV = os.environ.get('AN_SCEN'); OUT_PFX = os.environ.get('AN_OUT', 'results/')
def load_parts(sc, seed):
    P = {}
    for f in glob.glob(f"results/cache/{sc}_s{seed}_*.parquet"):
        P[f.split("_s%d_" % seed)[1][:-8]] = pd.read_parquet(f)
    return P

def add_ens(P):
    z = lambda x, mu, sd: (x - mu) / sd
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cal = P["cal_id"]
    mu1, sd1 = cal.lr.mean(), cal.lr.std(); l2 = lg(cal.lgbm.values); mu2, sd2 = l2.mean(), l2.std()
    for k, d in P.items():
        d["ens"] = z(d.lr, mu1, sd1) + z(lg(d.lgbm.values), mu2, sd2)
        d["lgbm_logit"] = lg(d.lgbm.values)
    return P

def sigm(x): return 1 / (1 + np.exp(-x))

def cwe_of(v):
    v = str(v)
    if v.startswith("["):
        try:
            l = ast.literal_eval(v); return l[0] if l else "none"
        except Exception: return "none"
    return v if v not in ("nan", "<NA>", "None") else "none"

def draw_thresholds(cal_pos, alpha, n, R, rng):
    """thresholds for R random subsamples of size n of cal_pos"""
    th = np.empty(R)
    for r in range(R):
        sub = rng.choice(cal_pos, min(n, len(cal_pos)), replace=False)
        th[r] = threshold_for_alpha(sub, alpha)
    return th

def draw_thresholds_pac(cal_pos, alpha, n, R, rng, delta=0.1):
    th = np.empty(R)
    for r in range(R):
        sub = rng.choice(cal_pos, min(n, len(cal_pos)), replace=False); th[r] = threshold_pac(sub, alpha, delta)
    return th

def stats_for_thr(th, s_sorted_pos, s_sorted_all):
    miss = np.searchsorted(s_sorted_pos, th, side="right") / len(s_sorted_pos)
    clear = np.searchsorted(s_sorted_all, th, side="right") / len(s_sorted_all)
    return miss, clear

def run(models=("lr", "svm", "lgbm", "ens")):
    det, cal_rows, tri, flag, buck, cwes, cost = [], [], [], [], [], [], []
    scen = sorted({f.split("/")[-1].split("_s")[0] for f in glob.glob("results/cache/*_cal_id.parquet")})
    scen = [s for s in scen if (s in SCEN_ENV.split(",") if SCEN_ENV else s in ("BV-R","BV-P","BV-T","DV-R","DV-P","DV2BV","BV2DV"))]
    for sc in scen:
        for seed in range(3):
            P = load_parts(sc, seed)
            if "cal_id" not in P or "test" not in P: continue
            P = add_ens(P)
            cal, te = P["cal_id"], P["test"]
            yc, yt = cal.y.values, te.y.values
            for m in models:
                sc_c, sc_t = cal[m].values, te[m].values
                thr = best_f1_threshold(sc_c, yc)
                d_te = detection_metrics(sc_t, yt, thr); d_cal = detection_metrics(sc_c, yc, thr)
                det.append(dict(scenario=sc, seed=seed, model=m, n_test=len(te), pos_test=int(yt.sum()),
                                **{f"test_{k}": v for k, v in d_te.items()}, **{f"id_{k}": v for k, v in d_cal.items()}))
                if "test_leaky" in P:
                    tl = P["test_leaky"]; dl = detection_metrics(tl[m].values, tl.y.values, thr)
                    det[-1].update({f"leaky_{k}": v for k, v in dl.items()}); det[-1]["n_leaky"] = len(tl)
                # ---- calibration (probabilities)
                if m in ("lr", "svm", "lgbm"):
                    raw_c = sigm(sc_c) if m == "lr" else (cal.lgbm.values if m == "lgbm" else None)
                    raw_t = sigm(sc_t) if m == "lr" else (te.lgbm.values if m == "lgbm" else None)
                    meths = ["platt", "isotonic"] + (["raw"] if raw_t is not None else [])
                    for meth in meths:
                        if meth == "raw": p = raw_t
                        elif m == "lgbm": p = to_prob(cal.lgbm_logit.values, yc, te.lgbm_logit.values, meth)
                        else: p = to_prob(sc_c, yc, sc_t, meth)
                        p = np.clip(p, 0, 1)
                        pc = None
                        cal_rows.append(dict(scenario=sc, seed=seed, model=m, method=meth, ece=ece(p, yt), brier=brier_score_loss(yt, p),
                                             mean_p=float(p.mean()), base=float(yt.mean())))
                # ---- triage guarantee
                cal_pos = sc_c[yc == 1]
                order_t = np.sort(sc_t); pos_t = np.sort(sc_t[yt == 1])
                has_w = "w" in cal.columns and cal.w.notna().any()
                for a in ALPHAS:
                    th = threshold_for_alpha(cal_pos, a)
                    miss, clear = stats_for_thr(np.array([th]), pos_t, order_t)
                    tri.append(dict(scenario=sc, seed=seed, model=m, alpha=a, regime="ID-all", miss=miss[0], clear=clear[0], viol=float(miss[0] > a), ncal=len(cal_pos),
                                    ks=ks_2samp(cal_pos, sc_t[yt == 1]).statistic))
                    th = draw_thresholds(cal_pos, a, NCAL, R, rng); miss, clear = stats_for_thr(th, pos_t, order_t)
                    tri.append(dict(scenario=sc, seed=seed, model=m, alpha=a, regime="ID", miss=miss.mean(), clear=clear.mean(), viol=float((miss > a).mean()), viol2=float((miss > a + 0.02).mean()), ncal=NCAL))
                    th = draw_thresholds_pac(cal_pos, a, NCAL, R, rng); miss, clear = stats_for_thr(th, pos_t, order_t)
                    tri.append(dict(scenario=sc, seed=seed, model=m, alpha=a, regime="ID-PAC", miss=miss.mean(), clear=clear.mean(), viol=float((miss > a).mean()), viol2=float((miss > a + 0.02).mean()), ncal=NCAL))
                    # weighted conformal
                    if has_w:
                        wc = cal.w.values[yc == 1]; wt = te.w.values; sc_pos = cal_pos
                        ms, cs = [], []
                        for r in range(60):
                            ix = rng.choice(len(sc_pos), min(NCAL, len(sc_pos)), replace=False)
                            p = pvalues_pos_w(sc_pos[ix], wc[ix], sc_t, wt)
                            ok = p <= a
                            ms.append(ok[yt == 1].mean()); cs.append(ok.mean())
                        ms = np.array(ms)
                        tri.append(dict(scenario=sc, seed=seed, model=m, alpha=a, regime="Weighted", miss=ms.mean(), clear=np.mean(cs), viol=float((ms > a).mean()), viol2=float((ms > a + 0.02).mean()), ncal=NCAL))
                    # target-labelled recalibration
                    if "cal_tgt" in P:
                        tg = P["cal_tgt"]; tpos = tg[m].values[tg.y.values == 1]
                        for k in KS + ["all"]:
                            kk = len(tpos) if k == "all" else k
                            if kk > len(tpos): continue
                            th = draw_thresholds(tpos, a, kk, R, rng); miss, clear = stats_for_thr(th, pos_t, order_t)
                            thp = draw_thresholds_pac(tpos, a, kk, R, rng); missp, clearp = stats_for_thr(thp, pos_t, order_t)
                            tri.append(dict(scenario=sc, seed=seed, model=m, alpha=a, regime=f"TargetPAC-{k}", miss=missp.mean(), clear=clearp.mean(), viol=float((missp > a).mean()), viol2=float((missp > a + 0.02).mean()), ncal=kk))
                            tri.append(dict(scenario=sc, seed=seed, model=m, alpha=a, regime=f"Target-{k}", miss=miss.mean(), clear=clear.mean(), viol=float((miss > a).mean()), viol2=float((miss > a + 0.02).mean()), ncal=kk))
                # ---- cost-optimal alpha (review cost 1 per fn, missed vulnerability costs c_m)
                grid = np.r_[np.linspace(0.01, 0.2, 20), np.linspace(0.25, 0.9, 14)]
                cal_sorted = np.sort(sc_c); pi_t = yt.mean()
                clr_cal = np.array([np.searchsorted(cal_sorted, threshold_for_alpha(cal_pos, g), side="right") / len(cal_sorted) for g in grid])
                for cm in (10, 50, 200, 1000):
                    nominal = (1 - clr_cal) + cm * yc.mean() * grid
                    gi = int(np.argmin(nominal)); a_star = grid[gi]
                    th = threshold_for_alpha(cal_pos, a_star)
                    miss_t, clear_t = stats_for_thr(np.array([th]), pos_t, order_t)
                    cost.append(dict(scenario=sc, seed=seed, model=m, c_m=cm, alpha_star=a_star, clear=clear_t[0], miss=miss_t[0],
                                     cost_triage=(1 - clear_t[0]) + cm * pi_t * miss_t[0], cost_all=1.0, cost_none=cm * pi_t))
                # ---- certified auto-flag
                for beta in (0.3, 0.5, 0.7):
                    lam = flag_threshold(sc_c, yc, beta)
                    fl = sc_t >= lam
                    flag.append(dict(scenario=sc, seed=seed, model=m, beta=beta, lam=lam, flagged=float(fl.mean()),
                                     prec=float(yt[fl].mean()) if fl.any() else np.nan, n_flag=int(fl.sum())))
                # ---- Mondrian (length buckets) vs marginal at alpha=0.1, using full calibration set
                edges = np.quantile(cal.length.values, [1 / 3, 2 / 3])
                bc = np.digitize(cal.length.values, edges); bt = np.digitize(te.length.values, edges)
                for a in (0.05, 0.10):
                    pm = auto_pass_mask(cal_pos, sc_t, a)
                    mp = mondrian_pass(sc_c[yc == 1], bc[yc == 1], sc_t, bt, a)
                    for name, mask in (("marginal", pm), ("mondrian", mp)):
                        for b in range(3):
                            sel = (bt == b) & (yt == 1)
                            buck.append(dict(scenario=sc, seed=seed, model=m, alpha=a, method=name, bucket=b, miss=float(mask[sel].mean()) if sel.any() else np.nan,
                                             clear=float(mask[bt == b].mean()), npos=int(sel.sum())))
                # ---- per-CWE
                if sc in ("DV-R", "BV-R") and m in ("lr", "lgbm"):
                    pm = auto_pass_mask(cal_pos, sc_t, 0.10)
                    cw = te.cwe.map(cwe_of)
                    for c, g in te.groupby(cw):
                        pos = (g.y.values == 1)
                        if pos.sum() >= 15:
                            cwes.append(dict(scenario=sc, seed=seed, model=m, cwe=c, npos=int(pos.sum()), miss=float(pm[g.index.values][pos].mean())))
            print(sc, seed, flush=True)
    pd.DataFrame(det).to_csv(OUT_PFX + "detection.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT_PFX + "calibration.csv", index=False)
    pd.DataFrame(tri).to_csv(OUT_PFX + "triage.csv", index=False)
    pd.DataFrame(flag).to_csv(OUT_PFX + "flag.csv", index=False)
    pd.DataFrame(buck).to_csv(OUT_PFX + "bucket.csv", index=False)
    pd.DataFrame(cwes).to_csv(OUT_PFX + "cwe.csv", index=False)
    pd.DataFrame(cost).to_csv(OUT_PFX + "cost.csv", index=False)

if __name__ == "__main__":
    run()
