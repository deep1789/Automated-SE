import glob, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("src/make_figs.py").read().split("det = pd.read_csv")[0])   # shared style / palette helpers
det = pd.read_csv("results/detection.csv")

# ---- leakage
fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.6), sharey=True)
for ax_, s in zip(axs, ["DV2BV", "BV2DV"]):
    g = det[(det.scenario == s)].groupby("model")[["test_auprc", "leaky_auprc"]].agg(["mean", "std"])
    for j, (col, lab, c) in enumerate((("test_auprc", "leak-free test", "#0072B2"), ("leaky_auprc", "leaked test functions", "#D55E00"))):
        ax_.bar(np.arange(4) + (j - .5) * .38, [g.loc[m, (col, "mean")] for m in ["lr", "svm", "lgbm", "ens"]], .38, yerr=[g.loc[m, (col, "std")] for m in ["lr", "svm", "lgbm", "ens"]], color=c, label=lab, error_kw=dict(lw=.6))
    ax_.set_xticks(range(4)); ax_.set_xticklabels([MN[m] for m in ["lr", "svm", "lgbm", "ens"]], rotation=25, ha="right"); ax_.set_title(SCN[s])
axs[0].set_ylabel("AUPRC"); axs[0].legend(fontsize=6.5); fig.tight_layout(); save(fig, "fig_leak")

# ---- Flawfinder operating curves
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
for ax_, s in zip(axs, ["DV-R", "BV-R", "BV-T", "DV2BV"]):
    d = pd.read_parquet(f"results/flaw_{s}.parquet"); y = d.y.values
    ml = d.lr.rank().values + d.lgbm.rank().values
    thr = np.quantile(ml[y == 1], np.linspace(0, 1, 200)); miss = np.array([(ml[y == 1] <= t).mean() for t in thr]); clr = np.array([(ml <= t).mean() for t in thr])
    ax_.plot(clr, miss, color=C["ens"], lw=1.4, label="learned scorer")
    ax_.plot([0, 1], [0, 1], color="#bbb", ls=":", lw=1, label="random clearing")
    cf = (d.flaw_level.values == 0); ax_.plot([0, cf.mean()], [0, cf[y == 1].mean()], "s-", color=C["flaw"], ms=4, lw=1, label="Flawfinder (no hit = clear)")
    ax_.axhline(.1, color="k", lw=.5, ls="--"); ax_.set_xlabel("share of functions cleared"); ax_.set_title(SCN[s])
axs[0].set_ylabel("share of vulnerable\nfunctions cleared (miss)"); axs[0].legend(fontsize=5.5, loc="upper left"); fig.tight_layout(); save(fig, "fig_flaw")

# ---- explanation
fid = pd.read_csv("results/explain_fidelity.csv"); imp = pd.read_csv("results/explain_importance.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
cols = {"TreeSHAP (local)": "#D55E00", "Global gain": "#0072B2", "Random": "#999999"}
for ax_, ds in zip(axs[:2], ["DiverseVul", "BigVul"]):
    for meth, c in cols.items():
        g = fid[(fid.ds == ds) & (fid.method == meth)].sort_values("k"); ax_.plot(g.k, g["drop"], "o-", ms=2.5, lw=1.1, color=c, label=meth)
    ax_.set_xlabel("top-$k$ features neutralised"); ax_.set_title("Deletion fidelity: " + ("DV" if ds == "DiverseVul" else "BV"))
axs[0].set_ylabel("mean drop in logit"); axs[0].legend(fontsize=6)
imp["m"] = (imp.DiverseVul + imp.BigVul) / 2; top = imp.sort_values("m", ascending=False).head(8).iloc[::-1]
y = np.arange(len(top)); axs[2].barh(y - .2, top.DiverseVul, .4, color="#0072B2", label="DV"); axs[2].barh(y + .2, top.BigVul, .4, color="#D55E00", label="BV")
axs[2].set_yticks(y); axs[2].set_yticklabels([f.replace("op_", "op ").replace("kw_", "kw ") for f in top.feature], fontsize=6.5); axs[2].set_xlabel("mean |SHAP|"); axs[2].legend(fontsize=6); axs[2].set_title("Global importance")
fig.tight_layout(); save(fig, "fig_explain")

# ---- robustness
fs = glob.glob("results/robust_*.csv")
if fs:
    r = pd.concat([pd.read_csv(f) for f in fs]); r = r[r["transform"] != "original"]
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    for ax_, sc in zip(axs, ["DV-R", "BV-R"]):
        for j, m in enumerate(["lr", "lgbm"]):
            g = r[(r.scenario == sc) & (r.model == m)].set_index("transform").loc[["rename", "deadcode", "both"]]
            ax_.bar(np.arange(3) + (j - .5) * .38, 100 * g.flip, .38, color=C[m], label=MN[m])
        ax_.set_xticks(range(3)); ax_.set_xticklabels(["rename\nlocals", "dead\ncode", "both"]); ax_.set_title(SCN[sc])
    axs[0].set_ylabel("clear/review decisions changed (%)"); axs[0].legend(fontsize=6.5); fig.tight_layout(); save(fig, "fig_robust")
print("ok")
