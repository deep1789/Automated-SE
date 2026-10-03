import json, glob, os
import numpy as np, pandas as pd
from scipy import stats
T = "paper/tables/"
SCN = {"DV-R": "DV random", "DV-P": "DV project", "BV-R": "BV random", "BV-P": "BV project", "BV-T": "BV temporal", "DV2BV": "DV$\\to$BV", "BV2DV": "BV$\\to$DV"}
ORDER = ["DV-R", "BV-R", "DV-P", "BV-P", "BV-T", "DV2BV", "BV2DV"]
MN = {"lr": "TF-IDF LR", "svm": "TF-IDF SVM", "lgbm": "LightGBM", "ens": "LR+LGBM"}
MODELS = ["lr", "svm", "lgbm", "ens"]
det = pd.read_csv("results/detection.csv"); tri = pd.read_csv("results/triage.csv"); cal = pd.read_csv("results/calibration.csv")
flag = pd.read_csv("results/flag.csv"); buck = pd.read_csv("results/bucket.csv"); cost = pd.read_csv("results/cost.csv")
SC = [s for s in ORDER if s in set(det.scenario)]
def w(name, s): open(T + name + ".tex", "w").write(s)
def pm(x, d=3, sd=True):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return "--"
    return f"{x.mean():.{d}f}" + (f"\\,{{\\scriptsize$\\pm${x.std(ddof=1) if len(x) > 1 else 0:.{d}f}}}" if sd else "")
def pct(x, d=1): x = np.asarray(x, float); x = x[~np.isnan(x)]; return "--" if len(x) == 0 else f"{100 * x.mean():.{d}f}"
def pctpm(x, d=1):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return "--" if len(x) == 0 else f"{100 * x.mean():.{d}f}" + f"{{\\scriptsize$\\pm${100 * (x.std(ddof=1) if len(x) > 1 else 0):.{d}f}}}"
def tab(cols, rows, caption, label, spec=None, note=None, size="\\small", star=False):
    spec = spec or ("l" + "c" * (len(cols) - 1))
    env = "table"; rs = "\\resizebox{\\linewidth}{!}{" if len(cols) >= 7 else ""
    s = f"\\begin{{{env}}}[t]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n{rs}\\begin{{tabular}}{{{spec}}}\n\\toprule\n" + " & ".join(cols) + " \\\\\n\\midrule\n"
    for r in rows: s += (r + "\n") if isinstance(r, str) else (" & ".join(r) + " \\\\\n")
    s += "\\bottomrule\n\\end{tabular}" + ("}" if len(cols) >= 7 else "") + "\n" + (f"\\\\[2pt]{{\\footnotesize {note}}}\n" if note else "") + f"\\end{{{env}}}\n"
    return s

# ---------------- data audit
a = json.load(open("results/audit.json"))
rows = []
for k, nm in (("DiverseVul", "DiverseVul"), ("BigVul", "Big-Vul")):
    d = a[k]
    rows.append([nm, f"{d['raw']:,}", f"{d['exact_dup_rows_removed']:,} ({100 * d['exact_dup_rows_removed'] / d['raw']:.1f}\\%)", f"{d['label_conflict_rows']:,}", f"{d['clean']:,}", f"{d['vuln']:,} ({100 * d['vuln_rate']:.1f}\\%)", f"{d['projects']:,}"])
w("tab_data", tab(["Dataset", "Raw", "Exact duplicates", "Label-conflict", "Clean", "Vulnerable", "Projects"], rows,
    f"Dataset audit after normalisation (comment/whitespace stripping). {a['cross_dataset_shared_functions']:,} normalised functions occur in both datasets: {100 * a['shared_frac_of_bigvul']:.1f}\\% of clean Big-Vul and {100 * a['shared_frac_of_diversevul']:.1f}\\% of clean DiverseVul.",
    "tab:data", "lrrrrrr", size="\\footnotesize"))

# ---------------- scenarios
desc = {"DV-R": "random 70/15/15", "BV-R": "random 70/15/15", "DV-P": "held-out projects (30\\%)", "BV-P": "held-out projects (30\\%)",
        "BV-T": "train $\\le$2015; test 2017--19", "DV2BV": "train DV, test BV (leak-free)", "BV2DV": "train BV, test DV (leak-free)"}
rows = []
for s in SC:
    inf = json.load(open(f"results/cache/{s}_s0_info.json"))
    pt = det[(det.scenario == s) & (det.seed == 0) & (det.model == "lr")].iloc[0]
    rows.append([SCN[s], desc[s], f"{inf['train_n']:,}", f"{inf['n_cal_id']:,}", f"{inf.get('n_cal_tgt', 0):,}" if inf.get("n_cal_tgt") else "--", f"{inf['n_test']:,}", f"{int(pt.pos_test):,}"])
w("tab_scen", tab(["Scenario", "Shift", "Train", "ID cal.", "Target pool", "Test", "Test pos."], rows, "Evaluation scenarios (seed 0 sizes; training keeps all vulnerable and at most 150\\,000 benign functions).", "tab:scen", "llrrrrr", size="\\footnotesize"))

# ---------------- detection
rows = []
for panel, met, nm in (("A", "test_auprc", "AUPRC"), ("B", "test_auroc", "AUROC"), ("C", "test_mcc", "MCC")):
    rows.append(f"\\midrule\n\\multicolumn{{5}}{{l}}{{\\emph{{Panel {panel}: {nm}}}}} \\\\")
    for s in SC:
        vals = {m: det[(det.scenario == s) & (det.model == m)][met].values for m in MODELS}
        best = max(MODELS, key=lambda m: vals[m].mean())
        rows.append([SCN[s]] + [("\\textbf{" + pm(vals[m]) + "}") if m == best else pm(vals[m]) for m in MODELS])
rows[0] = rows[0].replace("\\midrule\n", "")
w("tab_detect", tab(["Scenario"] + [MN[m] for m in MODELS], rows, "Detection performance on the held-out test sets (mean $\\pm$ s.d.\\ over 3 seeds; best per row in bold). MCC uses the F1-optimal threshold chosen on ID calibration data.", "tab:detect", "lcccc"))

# ---------------- leakage
rows = []
for s in [x for x in ("DV2BV", "BV2DV") if x in SC]:
    for m in ("lr", "lgbm"):
        g = det[(det.scenario == s) & (det.model == m)]
        rows.append([SCN[s], MN[m], f"{int(g.n_leaky.mean()):,}", pm(g.test_auprc), pm(g.leaky_auprc), pm(g.test_auroc), pm(g.leaky_auroc)])
w("tab_leak", tab(["Scenario", "Model", "Leaked $n$", "AUPRC clean", "AUPRC leaked", "AUROC clean", "AUROC leaked"], rows,
    "Effect of train--test leakage in the cross-dataset setting: ``leaked'' test functions have a normalised hash that occurs in the training data.", "tab:leak", "llrcccc", size="\\footnotesize"))

# ---------------- calibration
rows = []
for s in SC:
    r = [SCN[s]]
    for m in ("lr", "lgbm"):
        for meth in ("raw", "platt", "isotonic"):
            g = cal[(cal.scenario == s) & (cal.model == m) & (cal.method == meth)]
            r.append(pm(g.ece, 3, sd=False))
    rows.append(r)
w("tab_calib", tab(["Scenario", "LR raw", "LR Platt", "LR isotonic", "LGBM raw", "LGBM Platt", "LGBM isotonic"], rows,
    "Expected calibration error (ECE, lower is better) on the test sets; Platt scaling and isotonic regression are fitted on ID calibration data. Mean over 3 seeds.", "tab:calib", "lcccccc", size="\\footnotesize"))

# ---------------- alpha sweep (ID calibration, all calibration positives)
M0 = "ens"
rows = []
for s in SC:
    r = [SCN[s]]
    for al in (0.02, 0.05, 0.10, 0.20):
        g = tri[(tri.scenario == s) & (tri.model == M0) & (tri.regime == "ID-all") & (tri.alpha == al)]
        r.append(pctpm(g.miss))
    rows.append(r)
w("tab_alpha", tab(["Scenario", "$\\alpha{=}2\\%$", "$\\alpha{=}5\\%$", "$\\alpha{=}10\\%$", "$\\alpha{=}20\\%$"], rows,
    "Realised miss rate (\\%, mean $\\pm$ s.d.\\ over seeds) of the conformal clearance rule calibrated on source-distribution (ID) data, for four nominal budgets $\\alpha$ (LR+LGBM scorer).", "tab:alpha", "lcccc"))

# ---------------- guarantee table at alpha = 0.10
rows = []
for s in SC:
    def get(reg, col):
        g = tri[(tri.scenario == s) & (tri.model == M0) & (tri.regime == reg) & (tri.alpha == 0.10)]; return g[col].values
    r = [SCN[s], pctpm(get("ID-all", "miss")), pct(get("ID-all", "clear"))]
    if len(get("Weighted", "miss")): r += [pct(get("Weighted", "miss")), pct(get("Weighted", "clear"))]
    else: r += ["--", "--"]
    if len(get("Target-100", "miss")): r += [pct(get("Target-100", "miss")), pct(get("Target-100", "clear")), pct(get("TargetPAC-100", "miss")), pct(get("TargetPAC-100", "clear")), pct(get("TargetPAC-100", "viol2"), 0)]
    else: r += ["--"] * 5
    rows.append(r)
w("tab_guar", tab(["Scenario", "ID miss", "ID SAR", "Wtd miss", "Wtd SAR", "$k{=}100$ miss", "$k{=}100$ SAR", "PAC miss", "PAC SAR", "PAC viol."], rows,
    "Conformal clearance at $\\alpha{=}10\\%$ (LR+LGBM scorer; all figures in \\%). ID: calibrated on source data; Wtd: covariate-shift weighted; $k{=}100$: calibrated on 100 labelled target positives; PAC: $k{=}100$ with the high-probability rule ($\\delta{=}0.1$) and its violation frequency $\\Pr(M>\\alpha+2\\text{pp})$. A valid rule has miss $\\lesssim10$.", "tab:guar", "lccccccccc", size="\\footnotesize", star=True))

# ---------------- flag certification
rows = []
for s in SC:
    r = [SCN[s]]
    for b in (0.3, 0.5, 0.7):
        g = flag[(flag.scenario == s) & (flag.model == M0) & (flag.beta == b)]
        r += [pct(g.flagged, 2), pct(g.prec, 0) if g.prec.notna().any() else "--"]
    rows.append(r)
w("tab_flag", tab(["Scenario", "$\\beta{=}.3$ flagged", "prec.", "$\\beta{=}.5$ flagged", "prec.", "$\\beta{=}.7$ flagged", "prec."], rows,
    "Certified auto-flagging (Learn-then-Test, $\\delta{=}0.05$): share of test functions flagged (\\%) and realised precision (\\%); ``--'' = no threshold could be certified.", "tab:flag", "lcccccc", size="\\footnotesize"))

# ---------------- bucket (Mondrian)
rows = []
for s in SC:
    r = [SCN[s]]
    for meth in ("marginal", "mondrian"):
        g = buck[(buck.scenario == s) & (buck.model == M0) & (buck.alpha == 0.10) & (buck.method == meth)]
        by = g.groupby("bucket").miss.mean()
        r += [pct(by.loc[0]), pct(by.loc[1]), pct(by.loc[2]), pct(g.groupby("seed").clear.mean())]
    rows.append(r)
w("tab_bucket", tab(["Scenario", "short", "medium", "long", "SAR", "short", "medium", "long", "SAR"], rows,
    "Miss rate (\\%) per function-length tercile at $\\alpha{=}10\\%$ for marginal (left) and Mondrian (right) calibration, with the overall share cleared (SAR). LR+LGBM scorer, ID calibration.", "tab:bucket", "lcccccccc", size="\\footnotesize"))

# ---------------- cost
rows = []
for s in SC:
    r = [SCN[s]]
    for cm in (10, 50, 200):
        g = cost[(cost.scenario == s) & (cost.model == M0) & (cost.c_m == cm)]
        r += [f"{g.alpha_star.mean():.2f}", pct(g.clear, 0), f"{g.cost_triage.mean():.2f}"]
    rows.append(r)
w("tab_cost", tab(["Scenario", "$\\alpha^\\star$", "SAR", "cost", "$\\alpha^\\star$", "SAR", "cost", "$\\alpha^\\star$", "SAR", "cost"], rows,
    "Cost-optimal miss budget $\\alpha^\\star$ chosen on ID calibration data, its realised SAR (\\%) and expected cost per function relative to reviewing everything ($=1$), for $c_m=10$, $50$, $200$ review units per missed vulnerability. LR+LGBM scorer.", "tab:cost", "lccccccccc", size="\\footnotesize", star=True))


# ---------------- Flawfinder comparison
from sklearn.metrics import roc_auc_score, average_precision_score
rows = []
for s in SC:
    f = f"results/flaw_{s}.parquet"
    if not os.path.exists(f): continue
    d = pd.read_parquet(f); y = d.y.values
    ml = d.lr.rank().values + d.lgbm.rank().values
    clr = (d.flaw_level.values == 0); miss_f = clr[y == 1].mean(); sar_f = clr.mean()
    # ML at the same clearance share, and at the same miss rate (ROC operating-point comparison)
    thr = np.quantile(ml, sar_f); miss_ml = (ml[y == 1] <= thr).mean()
    thr2 = np.quantile(ml[y == 1], miss_f); sar_ml = (ml <= thr2).mean()
    rows.append([SCN[s], f"{roc_auc_score(y, d.flaw):.3f}", f"{roc_auc_score(y, ml):.3f}", f"{average_precision_score(y, d.flaw):.3f}", f"{average_precision_score(y, ml):.3f}",
                 f"{100 * sar_f:.1f}", f"{100 * miss_f:.1f}", f"{100 * miss_ml:.1f}", f"{100 * sar_ml:.1f}"])
if rows:
    w("tab_flaw", tab(["Scenario", "AUROC FF", "AUROC ML", "AUPRC FF", "AUPRC ML", "FF SAR", "FF miss", "ML miss @SAR", "ML SAR @miss"], rows,
    "Flawfinder (FF) versus the learned scorer (LR+LGBM rank average) on identical test functions (seed 0). The only conservative FF policy is to clear functions without any hit; its SAR and miss rate (\%) are compared with the learned scorer's miss rate at the same SAR and its SAR at the same miss rate.", "tab:flaw", "lcccccccc", size="\\footnotesize", star=True))

# ---------------- statistics
out = {}
pivot = det.pivot_table(index=["scenario", "seed"], columns="model", values="test_auprc").dropna()
fr = stats.friedmanchisquare(*[pivot[m].values for m in MODELS]); out["friedman_chi2"] = float(fr.statistic); out["friedman_p"] = float(fr.pvalue); out["friedman_n"] = int(len(pivot))
out["mean_rank"] = {m: float(v) for m, v in pivot.rank(axis=1, ascending=False).mean().items()}
# AUROC vs SAR(0.05) across blocks
sar = tri[(tri.regime == "ID-all") & (tri.alpha == 0.05)].rename(columns={"clear": "sar05"})[["scenario", "seed", "model", "sar05"]]
j = det.merge(sar, on=["scenario", "seed", "model"])
out["spearman_auroc_sar05"] = float(stats.spearmanr(j.test_auroc, j.sar05)[0]); out["spearman_auprc_sar05"] = float(stats.spearmanr(j.test_auprc, j.sar05)[0])
# within-scenario: do models with equal AUROC differ in SAR05?
wsp = []
for s in SC:
    g = j[j.scenario == s].groupby("model")[["test_auroc", "sar05"]].mean(); wsp.append(dict(scenario=s, auroc_range=float(g.test_auroc.max() - g.test_auroc.min()), sar05_range=float(g.sar05.max() - g.sar05.min())))
out["auroc_sar_ranges"] = wsp
# validity: random vs shift, miss-alpha
v = tri[(tri.regime == "ID-all") & (tri.model == M0)].copy(); v["exc"] = v.miss - v.alpha
for grp, ss in (("random", ["DV-R", "BV-R"]), ("shift", [x for x in SC if x not in ("DV-R", "BV-R")])):
    g = v[v.scenario.isin(ss)]
    out[f"excess_{grp}_mean"] = float(g.exc.mean())
    out[f"wilcoxon_excess_{grp}_p"] = float(stats.wilcoxon(g.exc.values, alternative="greater").pvalue) if len(g) > 5 else None
    out[f"ratio_{grp}"] = float((g.miss / g.alpha).mean())
ks = tri[(tri.regime == "ID-all") & tri.ks.notna()]
out["ks_bound_violations"] = int(((ks.miss - ks.alpha) > ks.ks + 0.02).sum()); out["ks_bound_n"] = int(len(ks))
out["ks_excess_spearman"] = float(stats.spearmanr(ks.ks, ks.miss - ks.alpha)[0])

# ---------------- robustness table
import glob as _g
rb = pd.concat([pd.read_csv(f) for f in sorted(_g.glob("results/robust_*.csv"))]) if _g.glob("results/robust_*.csv") else None
if rb is not None:
    rows = []
    for sc_ in ("DV-R", "BV-R"):
        for m in ("lr", "lgbm"):
            g = rb[(rb.scenario == sc_) & (rb.model == m)].set_index("transform")
            for tr_, nm in (("original", "original"), ("rename", "rename locals"), ("deadcode", "dead code"), ("both", "both")):
                x = g.loc[tr_]
                rows.append([SCN[sc_], MN[m], nm, f"{x.auroc:.3f}", f"{100 * x.miss:.1f}", f"{100 * x.flip:.1f}", f"{100 * x.flip_pos:.1f}", f"{x.rank_corr:.3f}"])
    w("tab_robust", tab(["Scenario", "Model", "Rewrite", "AUROC", "Miss (\\%)", "Flipped (\\%)", "Flipped, vuln. (\\%)", "Spearman $\\rho$"], rows,
        "Consistency of triage decisions ($\\alpha{=}10\\%$, ID calibration) on 1\\,200 vulnerable and 4\\,000 benign random test functions after semantics-preserving rewrites. ``Flipped'' = share of all (resp.\\ vulnerable) functions whose clear/review decision changes.", "tab:robust", "lllccccc", size="\\footnotesize"))
# ---------------- explanation table
ep = pd.read_csv("results/explain_perf.csv"); es = pd.read_csv("results/explain_stability.csv")
rows = []
for ds in ("DiverseVul", "BigVul", "DiverseVul->BigVul", "BigVul->DiverseVul"):
    g = ep[ep.ds == ds]; rows.append([ds.replace("->", "$\\to$").replace("DiverseVul", "DV").replace("BigVul", "BV"), pm(g.auroc), pm(g.auprc)])
w("tab_explain", tab(["Train$\\to$test", "AUROC", "AUPRC"], rows, "Interpretable metrics-only LightGBM used for the explanation analysis (3 seeds); cross-dataset rows exclude leaked functions.", "tab:explain", "lcc", size="\\footnotesize"))
# ---------------- pairwise tests
pw = {}
for m in ("lr", "svm", "lgbm"):
    a_, b_ = pivot["ens"].values, pivot[m].values
    gt = sum((x > y) for x in a_ for y in b_); lt = sum((x < y) for x in a_ for y in b_)
    pw[m] = dict(p=float(stats.wilcoxon(a_, b_).pvalue), cliff=float((gt - lt) / (len(a_) * len(b_))), mean_diff=float((a_ - b_).mean()))
out['pairwise_vs_ens']=pw
json.dump(out, open("results/stats.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:2500])
