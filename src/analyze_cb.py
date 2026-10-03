"""Detection and triage analysis for the CodeBERT experiment (negatives subsampled -> weighted AUPRC; SAR via true prevalence)."""
import glob, sys, os, numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "norm"; CACHE = os.environ.get("CB_CACHE") or ("results/cb_cache" if MODE == "raw" else "results/cbn_cache"); OUT = "results/cb_" if MODE == "raw" else "results/cbn_"
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp
from triage import threshold_for_alpha
from analyze import draw_thresholds, draw_thresholds_pac, stats_for_thr
rng = np.random.RandomState(11)
PREV = {"DiverseVul": 17689 / 325138, "BigVul": 8055 / 162274}
TARGET_DS = {"DV-R": "DiverseVul", "DV-P": "DiverseVul", "BV-R": "BigVul", "BV-P": "BigVul", "BV-T": "BigVul", "DV2BV": "BigVul", "BV2DV": "DiverseVul"}
RAW = ["cb_lr", "cb_gb", "tf_lr", "tf_gb"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def add_scores(P):
    for d in P.values():
        d["cb_gb"] = lg(d.cb_gb.values); d["tf_gb"] = lg(d.tf_gb.values)
        if "fmt_gb" in d: d["fmt_gb"] = lg(d.fmt_gb.values)
    c = P["cal_id"]; mu = {m: (c[m].mean(), c[m].std()) for m in RAW + (["ft_cb"] if "ft_cb" in c else [])}
    for d in P.values():
        z = lambda m: (d[m] - mu[m][0]) / mu[m][1]
        d["cb_ens"] = z("cb_lr") + z("cb_gb"); d["tf_ens"] = z("tf_lr") + z("tf_gb"); d["all_ens"] = z("cb_lr") + z("cb_gb") + z("tf_lr") + z("tf_gb")
        if "ft_cb" in d: d["ft_tf_ens"] = z("ft_cb") + z("tf_lr") + z("tf_gb")
    return P
MODELS = RAW + ["cb_ens", "tf_ens", "all_ens"] + (["fmt_gb"] if MODE == "norm" else []) + ["ft_cb", "ft_tf_ens"]
det, tri = [], []
for sc in ["DV-R", "BV-R", "DV-P", "BV-P", "BV-T", "DV2BV", "BV2DV"]:
    for seed in range(3):
        fs = glob.glob(f"{CACHE}/{sc}_s{seed}_*.parquet")
        if not fs: continue
        P = {f.split(f"_s{seed}_")[1][:-8]: pd.read_parquet(f) for f in fs}; P = add_scores(P)
        cal, te = P["cal_id"], P["test"]; yc, yt = cal.y.values, te.y.values; pi = PREV[TARGET_DS[sc]]
        for m in [m for m in MODELS if m in cal.columns]:
            sc_c, sc_t = cal[m].values, te[m].values
            det.append(dict(scenario=sc, seed=seed, model=m, auroc=roc_auc_score(yt, sc_t), auprc=average_precision_score(yt, sc_t, sample_weight=te.w.values),
                            id_auprc=average_precision_score(yc, sc_c, sample_weight=cal.w.values), n_pos=int(yt.sum())))
            cal_pos = sc_c[yc == 1]; neg_t = np.sort(sc_t[yt == 0]); pos_t = np.sort(sc_t[yt == 1])
            def row(regime, a, th, **kw):
                miss = np.searchsorted(pos_t, th, side="right") / len(pos_t); tnr = np.searchsorted(neg_t, th, side="right") / len(neg_t)
                return dict(scenario=sc, seed=seed, model=m, alpha=a, regime=regime, miss=np.mean(miss), sar=np.mean((1 - pi) * tnr + pi * miss), **kw)
            for a in (0.05, 0.10, 0.20):
                tri.append(row("ID-all", a, np.array([threshold_for_alpha(cal_pos, a)]), ks=ks_2samp(cal_pos, sc_t[yt == 1]).statistic))
                th = draw_thresholds(cal_pos, a, 300, 200, rng); r = row("ID", a, th); r["viol2"] = float((np.searchsorted(pos_t, th, side="right") / len(pos_t) > a + 0.02).mean()); tri.append(r)
                if "cal_tgt" in P:
                    tg = P["cal_tgt"]; tpos = tg[m].values[tg.y.values == 1]
                    for k in (50, 100, 200):
                        if k > len(tpos): continue
                        for reg, fn in (("Target", draw_thresholds), ("TargetPAC", draw_thresholds_pac)):
                            th = fn(tpos, a, k, 200, rng); r = row(f"{reg}-{k}", a, th); r["viol2"] = float((np.searchsorted(pos_t, th, side="right") / len(pos_t) > a + 0.02).mean()); tri.append(r)
    print(sc, flush=True)
pd.DataFrame(det).to_csv(OUT + "detection.csv", index=False); pd.DataFrame(tri).to_csv(OUT + "triage.csv", index=False)
