import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False, "axes.titlesize": 9, "axes.labelsize": 8.5})
det, tri = pd.read_csv("results/pv_detection.csv"), pd.read_csv("results/pv_triage.csv")
SC = ["PV-R", "PV-P", "PV-T", "DV2PV", "BV2PV", "PV2DV", "PV2BV"]; NM = {"PV-R": "PV random", "PV-P": "PV project", "PV-T": "PV temporal", "DV2PV": "DV$\\to$PV", "BV2PV": "BV$\\to$PV", "PV2DV": "PV$\\to$DV", "PV2BV": "PV$\\to$BV"}
def g(s, reg, col, a=0.10, m="ens"):
    d = tri[(tri.scenario == s) & (tri.model == m) & (tri.regime == reg) & (tri.alpha == a)]; return d[col].mean() if len(d) else np.nan
def f(x, d=1, p=True): return "--" if np.isnan(x) else (f"{100 * x:.{d}f}" if p else f"{x:.{d}f}")
rows = []
for s in SC:
    de = det[(det.scenario == s) & (det.model == "ens")]
    rows.append([NM[s], f(de.test_auroc.mean(), 3, False), f(de.test_auprc.mean(), 3, False), f(g(s, "ID-all", "miss")), f(g(s, "ID-all", "clear")), f(g(s, "Weighted", "miss")), f(g(s, "Target-100", "miss")), f(g(s, "Target-100", "clear")), f(g(s, "TargetPAC-100", "clear")), f(g(s, "TargetPAC-100", "viol2"), 1)])
s_ = "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Replication on PrimeVul (LR+LGBM scorer, mean over 3 seeds; \\%, except AUROC/AUPRC). ID: source-calibrated clearance at $\\alpha{=}10\\%$; Wtd: covariate-shift weighted; $k{=}100$: calibrated on 100 labelled target positives; PAC: high-probability variant with its violation frequency. Cross-dataset scenarios exclude target functions that occur in the training corpus.}\\label{tab:pv}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccccccc}\n\\toprule\nScenario & AUROC & AUPRC & ID miss & ID SAR & Wtd miss & $k{=}100$ miss & $k{=}100$ SAR & PAC SAR & PAC viol. \\\\\n\\midrule\n"
for r in rows: s_ += " & ".join(r) + " \\\\\n"
open("paper/tables/tab_pv.tex", "w").write(s_ + "\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
a = json.load(open("results/audit_pv.json")); b = json.load(open("results/audit.json"))
rows = [[n, f"{d['raw']:,}", f"{d['exact_dup_rows_removed']:,}", f"{d['clean']:,}", f"{d['vuln']:,} ({100 * d['vuln_rate']:.1f}\\%)", f"{d['projects']:,}"] for n, d in (("DiverseVul", b["DiverseVul"]), ("Big-Vul", b["BigVul"]), ("PrimeVul", a))]
s_ = "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Three corpora after normalisation and de-duplication. Pairwise overlaps of normalised functions: " + f"DiverseVul$\\cap$Big-Vul {b['cross_dataset_shared_functions']:,} ({100 * b['shared_frac_of_bigvul']:.1f}\\% of Big-Vul), PrimeVul$\\cap$DiverseVul {a['shared_with_DiverseVul']:,} ({100 * a['shared_frac_of_PrimeVul_DiverseVul']:.1f}\\% of PrimeVul), PrimeVul$\\cap$Big-Vul {a['shared_with_BigVul']:,} ({100 * a['shared_frac_of_PrimeVul_BigVul']:.1f}\\% of PrimeVul)." + "}\\label{tab:data3}\n\\begin{tabular}{lrrrrr}\n\\toprule\nCorpus & Raw & Exact duplicates & Clean & Vulnerable & Projects \\\\\n\\midrule\n"
for r in rows: s_ += " & ".join(r) + " \\\\\n"
open("paper/tables/tab_data3.tex", "w").write(s_ + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
fa = json.load(open("results/format_ablation.json")); cols = ["all", "only whitespace structure", "only size (chars, lines, mean line length)", "only comment markers", "all except whitespace structure"]
rows = [[nm] + [f"{fa[ds][c]:.3f}" for c in cols] for ds, nm in (("BigVul", "Big-Vul"), ("DiverseVul", "DiverseVul"), ("PrimeVul", "PrimeVul"))]
s_ = "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Which formatting statistics carry the signal? AUROC of a LightGBM on groups of 15 raw-text statistics (random 70/30 split of a stratified sample): all, whitespace structure alone (tabs, carriage returns, trailing whitespace, leading whitespace, indentation, brace placement), size alone (characters, lines, mean line length), comment markers alone, and everything except whitespace structure.}\\label{tab:formatabl}\n\\begin{tabular}{lccccc}\n\\toprule\nCorpus & All & Whitespace only & Size only & Comments only & All but whitespace \\\\\n\\midrule\n"
for r in rows: s_ += " & ".join(r) + " \\\\\n"
open("paper/tables/tab_formatabl.tex", "w").write(s_ + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# figure
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7)); cm = plt.cm.tab10
axs[0].plot([0, .22], [0, .22], "k--", lw=.8)
for i, s in enumerate(SC):
    d = tri[(tri.scenario == s) & (tri.model == "ens") & (tri.regime == "ID-all")].groupby("alpha").miss.agg(["mean", "std"]); axs[0].errorbar(d.index, d["mean"], d["std"], marker="o", ms=3, lw=1, capsize=2, color=cm(i), label=NM[s].replace("$\\to$", "→"))
axs[0].set_xlabel("target miss rate $\\alpha$"); axs[0].set_ylabel("realised miss rate on test"); axs[0].legend(fontsize=6); axs[0].set_title("PrimeVul: source-calibrated clearance")
w = .2; regs = [("ID-all", "source-calibrated", "#999999"), ("Weighted", "weighted", "#CC79A7"), ("Target-100", "100 target positives", "#D55E00"), ("TargetPAC-100", "PAC, 100 target positives", "#009E73")]
for j, (r, lab, c) in enumerate(regs): axs[1].bar(np.arange(len(SC)) + (j - 1.5) * w, [100 * (g(s, r, "miss") if not np.isnan(g(s, r, "miss")) else 0) for s in SC], w, color=c, label=lab)
axs[1].axhline(10, color="k", ls="--", lw=.8); axs[1].set_xticks(range(len(SC))); axs[1].set_xticklabels([NM[s].replace("$\\to$", "→") for s in SC], rotation=35, ha="right"); axs[1].set_ylabel("miss rate (%) at $\\alpha{=}10\\%$"); axs[1].legend(fontsize=5.8)
fig.tight_layout(); fig.savefig("paper/figs/fig_pv.pdf"); fig.savefig("paper/figs/fig_pv.png", dpi=200)
k = tri[(tri.regime == "ID-all") & tri.ks.notna()]; x = tri[(tri.regime == "ID-all") & (tri.alpha == .1) & (tri.model == "ens")]
print(json.dumps(dict(ks_viol=int(((k.miss - k.alpha) > k.ks).sum()), ks_n=len(k), rand=float(x[x.scenario == "PV-R"].miss.mean()), shift=float(x[x.scenario != "PV-R"].miss.mean())), indent=1))
