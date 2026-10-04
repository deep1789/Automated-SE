import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as Beta
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False, "axes.titlesize": 9, "axes.labelsize": 8.5})
C = {"lr": "#0072B2", "svm": "#56B4E9", "lgbm": "#D55E00", "ens": "#009E73", "flaw": "#7f7f7f"}
MN = {"lr": "TF-IDF LR", "svm": "TF-IDF SVM", "lgbm": "LightGBM", "ens": "LR+LGBM"}
SCN = {"DV-R": "DV random", "DV-P": "DV project", "BV-R": "BV random", "BV-P": "BV project", "BV-T": "BV temporal", "DV2BV": "DV→BV", "BV2DV": "BV→DV"}
ORDER = ["DV-R", "BV-R", "DV-P", "BV-P", "BV-T", "DV2BV", "BV2DV"]
OUT = "paper/figs/"
def save(fig, name): fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=200); plt.close(fig)

det = pd.read_csv("results/detection.csv"); tri = pd.read_csv("results/triage.csv"); cal = pd.read_csv("results/calibration.csv")
buck = pd.read_csv("results/bucket.csv"); cwe = pd.read_csv("results/cwe.csv"); cost = pd.read_csv("results/cost.csv")
sc_present = [s for s in ORDER if s in set(det.scenario)]

# ---- F2: detection performance across scenarios
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
for a, met, ttl in zip(ax, ["test_auroc", "test_auprc"], ["AUROC", "AUPRC"]):
    w = 0.2
    for j, m in enumerate(["lr", "svm", "lgbm", "ens"]):
        g = det[det.model == m].groupby("scenario")[met].agg(["mean", "std"]).reindex(sc_present)
        a.bar(np.arange(len(sc_present)) + (j - 1.5) * w, g["mean"], w, yerr=g["std"], color=C[m], label=MN[m], error_kw=dict(lw=.6))
    a.set_xticks(range(len(sc_present))); a.set_xticklabels([SCN[s] for s in sc_present], rotation=35, ha="right"); a.set_ylabel(ttl); a.set_title(ttl + " on the held-out test set")
ax[0].axhline(.5, color="k", lw=.6, ls=":"); ax[0].legend(ncol=2, fontsize=7, loc="lower left")
fig.tight_layout(); save(fig, "fig_detection")

# ---- F4: realised miss rate vs nominal alpha (marginal guarantee), ID calibration
t = tri[(tri.regime == "ID")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
mdl = "ens"
for ax_, ms, ttl in zip(axs, [["DV-R", "BV-R"], [s for s in ["DV-P", "BV-P", "BV-T", "DV2BV", "BV2DV"] if s in sc_present]], ["Exchangeable splits (guarantee should hold)", "Shifted splits"]):
    ax_.plot([0, .22], [0, .22], "k--", lw=.8, label="nominal $\\alpha$")
    for i, s in enumerate(ms):
        g = t[(t.scenario == s) & (t.model == mdl)].groupby("alpha").miss.agg(["mean", "std"])
        if len(g): ax_.errorbar(g.index, g["mean"], g["std"], marker="o", ms=3, lw=1, capsize=2, label=SCN[s], color=plt.cm.tab10(i if ms[0] == "DV-R" else i + 2))
    ax_.set_xlabel("target miss rate $\\alpha$"); ax_.set_ylabel("realised miss rate on test"); ax_.set_title(ttl); ax_.legend(fontsize=6.5)
fig.tight_layout(); save(fig, "fig_validity")

# ---- F5: excess miss vs KS distance
d = tri[tri.regime == "ID-all"].dropna(subset=["ks"]).copy(); d["excess"] = d.miss - d.alpha
fig, ax = plt.subplots(figsize=(3.6, 3.1))
for s in sc_present:
    g = d[d.scenario == s]; ax.scatter(g.ks, g.excess, s=9, alpha=.6, label=SCN[s])
xs = np.linspace(0, d.ks.max() * 1.05, 10); ax.plot(xs, xs, "k--", lw=.8, label="bound: excess $=$ KS")
ax.axhline(0, color="gray", lw=.5); ax.set_xlabel("KS distance, calibration vs test positives"); ax.set_ylabel("miss rate $-\\ \\alpha$")
ax.legend(fontsize=5.5, ncol=2); fig.tight_layout(); save(fig, "fig_ks_bound")

# ---- F6: recalibration with k labelled target positives
shift = [s for s in ["DV-P", "BV-P", "BV-T", "DV2BV", "BV2DV"] if s in sc_present]
fig, axs = plt.subplots(1, len(shift), figsize=(1.9 * len(shift) + .6, 2.5), sharey=True)
axs = np.atleast_1d(axs)
for ax_, s in zip(axs, shift):
    g = tri[(tri.scenario == s) & (tri.alpha == .1) & (tri.model == mdl)]
    ks_ = [25, 50, 100, 200, 400]
    mm = [g[g.regime == f"Target-{k}"].miss.mean() for k in ks_]
    ax_.plot(ks_, mm, "o-", color=C["lgbm"], ms=3, label="labelled target positives")
    for reg, col, ls, lab in (("ID", "#444", "--", "source-calibrated"), ("Weighted", "#CC79A7", ":", "covariate-shift weighted")):
        v = g[g.regime == reg].miss.mean(); ax_.axhline(v, color=col, ls=ls, lw=1, label=lab)
    kk = np.array([25, 50, 100, 200, 400]); lo = Beta.ppf(.05, np.floor(.1 * (kk + 1)), kk + 1 - np.floor(.1 * (kk + 1))); hi = Beta.ppf(.95, np.floor(.1 * (kk + 1)), kk + 1 - np.floor(.1 * (kk + 1)))
    ax_.fill_between(kk, lo, hi, color="gray", alpha=.2, label="theory: 90% band (exchangeable)")
    ax_.axhline(.1, color="k", lw=.7); ax_.set_xscale("log"); ax_.set_xticks([25, 100, 400]); ax_.set_xticklabels(["25", "100", "400"]); ax_.minorticks_off(); ax_.set_xlabel("$k$ labelled positives"); ax_.set_title(SCN[s])
axs[0].set_ylabel("realised miss rate ($\\alpha{=}0.10$)"); axs[0].legend(fontsize=5.5, loc="upper right")
fig.tight_layout(); save(fig, "fig_recalibration")

# ---- F7: clearance (effort saved) vs alpha
fig, ax = plt.subplots(figsize=(4.2, 2.9))
for i, s in enumerate(sc_present):
    g = tri[(tri.scenario == s) & (tri.model == "ens") & (tri.regime == "ID-all")].groupby("alpha").clear.mean()
    ax.plot(g.index, g.values, "o-", ms=3, lw=1, label=SCN[s], color=plt.cm.tab10(i))
ax.set_xlabel("target miss rate $\\alpha$"); ax.set_ylabel("share of functions auto-cleared (SAR)"); ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); save(fig, "fig_sar")

# ---- F8: Mondrian
b = buck[(buck.alpha == .10) & buck.scenario.isin(["DV-R", "BV-R"]) & (buck.model.isin(["ens"]))]
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=True)
for ax_, s in zip(axs, ["DV-R", "BV-R"]):
    for j, (meth, col) in enumerate((("marginal", "#999"), ("mondrian", C["lgbm"]))):
        g = b[(b.scenario == s) & (b.method == meth)].groupby("bucket").miss.agg(["mean", "std"])
        ax_.bar(np.arange(3) + (j - .5) * .38, g["mean"], .38, yerr=g["std"], color=col, label=("marginal calibration" if meth == "marginal" else "length-Mondrian calibration"), error_kw=dict(lw=.6))
    ax_.axhline(.1, color="k", ls="--", lw=.8); ax_.set_xticks(range(3)); ax_.set_xticklabels(["short", "medium", "long"]); ax_.set_title(SCN[s]); ax_.set_xlabel("function-length tercile")
axs[0].set_ylabel("miss rate ($\\alpha{=}0.10$)"); axs[0].legend(fontsize=6)
fig.tight_layout(); save(fig, "fig_mondrian")

# ---- F9: per-CWE miss
fig, axs = plt.subplots(1, 2, figsize=(7, 3.0))
for ax_, s in zip(axs, ["DV-R", "BV-R"]):
    g = cwe[(cwe.scenario == s) & (cwe.model == "lgbm")].groupby("cwe").agg(miss=("miss", "mean"), npos=("npos", "mean")).sort_values("npos", ascending=False).head(12).sort_values("miss")
    ax_.barh(g.index, g.miss, color=C["lgbm"]); ax_.axvline(.1, color="k", ls="--", lw=.8); ax_.set_title(SCN[s] + " (LightGBM, $\\alpha{=}0.10$)"); ax_.set_xlabel("miss rate per CWE")
fig.tight_layout(); save(fig, "fig_cwe")

# ---- F10: cost
fig, ax = plt.subplots(figsize=(4.4, 2.9))
cc = cost[cost.model == "ens"]
for i, s in enumerate(sc_present):
    g = cc[cc.scenario == s].groupby("c_m").cost_triage.mean(); ax.plot(g.index, g.values, "o-", ms=3, lw=1, color=plt.cm.tab10(i), label=SCN[s])
ax.axhline(1, color="k", ls="--", lw=.8); ax.text(11, 1.03, "review everything", fontsize=6.5); ax.set_xscale("log"); ax.set_xlabel("cost of a missed vulnerability / cost of one review"); ax.set_ylabel("expected cost per function\n(relative to review-all)")
ax.legend(fontsize=6, ncol=2); fig.tight_layout(); save(fig, "fig_cost")
print("figs done")
