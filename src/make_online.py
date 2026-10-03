import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False, "axes.titlesize": 9, "axes.labelsize": 8.5})
o = pd.read_csv("results/online.csv"); SC = ["BV-T", "BV-P", "DV-P", "DV2BV", "BV2DV", "DV-R", "BV-R"]
NM = {"BV-T": "BV temporal", "BV-P": "BV project", "DV-P": "DV project", "DV2BV": "DV$\\to$BV", "BV2DV": "BV$\\to$DV", "DV-R": "DV random", "BV-R": "BV random"}
METH = [("static", None, "Static (source-calibrated)"), ("sw", None, "Sliding window ($W{=}200$)"), ("aci", 0.01, "ACI ($\\gamma{=}0.01$)"), ("aci_sw", 0.02, "ACI + window")]
def g(sc, m, gm, col, a=0.10):
    d = o[(o.scenario == sc) & (o.method == m) & (o.alpha == a)]
    if gm is not None: d = d[d.gamma == gm]
    return d[col].mean()
# table
rows = []
for sc in SC:
    r = [NM[sc]]
    for m, gm, _ in METH: r.append(f"{100 * g(sc, m, gm, 'miss'):.1f}")
    for m, gm, _ in METH: r.append(f"{100 * g(sc, m, gm, 'worst_win'):.1f}")
    for m, gm, _ in METH: r.append(f"{100 * g(sc, m, gm, 'sar'):.1f}")
    rows.append(r)
s = "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Online risk control at $\\alpha{=}10\\%$ on arrival-ordered test streams (batches of 1\\,000 functions, labels revealed after each batch; LR+LGBM scorer; mean over 3 seeds, \\%). Overall miss rate over the stream, worst miss rate in any window of 300 vulnerable functions, and share of functions cleared (SAR). Static: threshold fixed from source calibration; SW: recalibrated on the last 200 observed vulnerable functions; ACI: adaptive conformal inference with step $\\gamma$; ACI+window: ACI on the sliding-window reference.}\\label{tab:online}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * 12 + "}\n\\toprule\n"
s += "& \\multicolumn{4}{c}{overall miss} & \\multicolumn{4}{c}{worst window miss} & \\multicolumn{4}{c}{SAR} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\nStream" + " & Static & SW & ACI & ACI+W" * 3 + " \\\\\n\\midrule\n"
for r in rows: s += " & ".join(r) + " \\\\\n"
open("paper/tables/tab_online.tex", "w").write(s + "\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
# gamma sensitivity table
rows = []
for sc in ("BV-T", "BV-P", "DV2BV"):
    for gm in (0.005, 0.01, 0.02, 0.05):
        a = o[(o.scenario == sc) & (o.method == "aci") & (o.gamma == gm) & (o.alpha == .10)]
        rows.append([NM[sc], f"{gm}", f"{100 * a.miss.mean():.1f}", f"{100 * a.bound.mean():.1f}", f"{100 * a.sar.mean():.1f}", f"{100 * a.worst_win.mean():.1f}"])
s = "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Sensitivity of ACI to the step size $\\gamma$ ($\\alpha{=}10\\%$): overall miss, the deterministic bound $(1+\\gamma b)/(\\gamma N)$ of Proposition~\\ref{prop:online} on its deviation from $\\alpha$ (\\%), SAR and worst-window miss.}\\label{tab:gamma}\n\\begin{tabular}{lccccc}\n\\toprule\nStream & $\\gamma$ & miss & bound & SAR & worst window \\\\\n\\midrule\n" + "".join(" & ".join(r) + " \\\\\n" for r in rows)
open("paper/tables/tab_gamma.tex", "w").write(s + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# figure
tr = json.load(open("results/online_traj.json")); fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7))
C = {"static": "#999999", "sw": "#0072B2", "aci": "#D55E00", "aci_sw": "#009E73"}
for m, gm, lab in METH:
    if m in tr: v = np.array(tr[m], float); axs[0].plot(np.arange(len(v)), 100 * v, color=C[m], lw=1.1, label=lab)
axs[0].axhline(10, color="k", ls="--", lw=.8); axs[0].set_xlabel("vulnerable functions seen (stream order)"); axs[0].set_ylabel("miss rate in last 300 (%)"); axs[0].set_title("Big-Vul temporal stream"); axs[0].legend(fontsize=6)
w = 0.2
for j, (m, gm, lab) in enumerate(METH):
    axs[1].bar(np.arange(len(SC)) + (j - 1.5) * w, [100 * g(sc, m, gm, "miss") for sc in SC], w, color=C[m])
axs[1].axhline(10, color="k", ls="--", lw=.8); axs[1].set_xticks(range(len(SC))); axs[1].set_xticklabels([NM[s].replace("$\\to$", "→") for s in SC], rotation=35, ha="right"); axs[1].set_ylabel("overall miss rate (%)"); axs[1].set_title("Whole stream, $\\alpha{=}10\\%$")
fig.tight_layout(); fig.savefig("paper/figs/fig_online.pdf"); fig.savefig("paper/figs/fig_online.png", dpi=200)
a = o[o.method == "aci"]; print("ACI bound holds", int(((a.miss - a.alpha).abs() <= a.bound).sum()), "/", len(a))
