import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False, "axes.titlesize": 9, "axes.labelsize": 8.5})
SCN = {"DV-R": "DV random", "DV-P": "DV project", "BV-R": "BV random", "BV-P": "BV project", "BV-T": "BV temporal", "DV2BV": "DV$\\to$BV", "BV2DV": "BV$\\to$DV"}
ORDER = ["DV-R", "BV-R", "DV-P", "BV-P", "BV-T", "DV2BV", "BV2DV"]
dn, tn = pd.read_csv("results/cbn_detection.csv"), pd.read_csv("results/cbn_triage.csv"); dr, tr_ = pd.read_csv("results/cb_detection.csv"), pd.read_csv("results/cb_triage.csv")
COLS = [("fmt_gb", dn, tn, "Format-only"), ("tf_ens", dn, tn, "TF-IDF+LGBM"), ("cb_ens", dr, tr_, "CodeBERT (raw text)"), ("cb_ens", dn, tn, "CodeBERT (normalised)"), ("all_ens", dn, tn, "TF-IDF+CodeBERT (norm.)")]
CLR = ["#999999", "#0072B2", "#E69F00", "#D55E00", "#009E73"]
HAS_FT_ANY = (dn.model == "ft_cb").any(); HAS_FT = False   # main tables keep 3 seeds for all models; fine-tuned results get a matched-seed table below
if HAS_FT: COLS.append(("ft_cb", dn, tn, "CodeBERT (fine-tuned)")); CLR.append("#CC79A7")
NC = len(COLS)
def val(d, m, s, col): return d[(d.model == m) & (d.scenario == s)][col].mean()
def tex(cols, rows, cap, lab, spec):
    s = "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{" + cap + "}\\label{" + lab + "}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{" + spec + "}\n\\toprule\n" + " & ".join(cols) + " \\\\\n\\midrule\n"
    for r in rows: s += (r + "\n") if isinstance(r, str) else (" & ".join(r) + " \\\\\n")
    return s + "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
short = ["Fmt", "TF-IDF", "CB raw", "CB norm", "TF+CB"] + (["CB FT"] if HAS_FT else [])
rows = []
for s in ORDER:
    
    r = [SCN[s]]
    for c in ("auroc", "auprc"):
        for i, (m, d, t, _) in enumerate(COLS):
            v = f"{val(d, m, s, c):.3f}"; r.append(v)
    rows.append(r)
open("paper/tables/tab_cb_detect.tex", "w").write(tex(["Scenario"] + [f"AUROC {x}" for x in short] + [f"AUPRC {x}" for x in short], rows,
    "Detection with frozen CodeBERT embeddings on the stratified subsample (mean over 3 seeds; AUPRC weighted to the true prevalence). Fmt: formatting-only classifier; CB raw / norm: CodeBERT on raw text / on comment-and-whitespace-normalised text; ensembles are standardised score sums.", "tab:cbdetect", "l" + "c" * (2 * NC)))
def tri(m, d, s, reg, col, a=0.10): return d[(d.model == m) & (d.scenario == s) & (d.regime == reg) & (d.alpha == a)][col].mean()
rows = []
for s in ORDER:
    r = [SCN[s]] + [f"{100 * tri(m, t, s, 'ID-all', 'miss'):.1f}" for m, d, t, _ in COLS]
    r += [("--" if s in ("DV-R", "BV-R") else f"{100 * tri(m, t, s, 'Target-100', 'sar'):.1f}") for m, d, t, _ in COLS]
    rows.append(r)
rows.append("\\midrule")
sh = [s for s in ORDER if s not in ("DV-R", "BV-R")]
rows.append(["Mean, random"] + [f"{100 * np.mean([tri(m, t, s, 'ID-all', 'miss') for s in ('DV-R', 'BV-R')]):.1f}" for m, d, t, _ in COLS] + [""] * NC)
rows.append(["Mean, shifted"] + [f"{100 * np.mean([tri(m, t, s, 'ID-all', 'miss') for s in sh]):.1f}" for m, d, t, _ in COLS] + [f"{100 * np.mean([tri(m, t, s, 'Target-100', 'sar') for s in sh]):.1f}" for m, d, t, _ in COLS])
open("paper/tables/tab_cb_triage.tex", "w").write(tex(["Scenario"] + [f"miss {x}" for x in short] + [f"SAR {x}" for x in short], rows,
    "Triage with CodeBERT scorers at $\\alpha{=}10\\%$ (\\%). Left: realised miss rate when calibrated on source (ID) data; right: safe automation rate when calibrated on 100 labelled target positives (valid by Prop.~\\ref{prop:valid}). Subsample experiment; see Table~\\ref{tab:cbdetect} for column keys.", "tab:cbtriage", "l" + "c" * (2 * NC)))
# ---- figure
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), sharex=True)
w = 0.8 / NC
for j, (m, d, t, lab) in enumerate(COLS):
    axs[0].bar(np.arange(7) + (j - (NC - 1) / 2) * w, [val(d, m, s, "auroc") for s in ORDER], w, color=CLR[j], label=lab)
    axs[1].bar(np.arange(7) + (j - (NC - 1) / 2) * w, [100 * tri(m, t, s, "ID-all", "miss") for s in ORDER], w, color=CLR[j])
axs[0].set_ylim(.5, 1); axs[0].set_ylabel("AUROC"); axs[1].axhline(10, color="k", ls="--", lw=.8); axs[1].set_ylabel("realised miss rate (%) at $\\alpha{=}10\\%$")
for a in axs: a.set_xticks(range(7)); a.set_xticklabels([SCN[s].replace("$\\to$", "→") for s in ORDER], rotation=35, ha="right")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=min(NC, 6), fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.09)); axs[0].set_title("Detection (raw text exploits the Big-Vul artifact)"); axs[1].set_title("Source-calibrated triage")
fig.tight_layout(); fig.savefig("paper/figs/fig_cb.pdf"); fig.savefig("paper/figs/fig_cb.png", dpi=200)
# ---- numbers for the text
import json
ks = tn[(tn.regime == "ID-all") & tn.ks.notna()]; kr = tr_[(tr_.regime == "ID-all") & tr_.ks.notna()]
o = dict(ks_viol_norm=int(((ks.miss - ks.alpha) > ks.ks).sum()), ks_n_norm=len(ks), ks_viol_raw=int(((kr.miss - kr.alpha) > kr.ks).sum()), ks_n_raw=len(kr))
for nm, m, t in list((("ft", "ft_cb", tn),) if HAS_FT else ()) + [("tf", "tf_ens", tn), ("cbn", "cb_ens", tn), ("cbr", "cb_ens", tr_), ("all", "all_ens", tn), ("fmt", "fmt_gb", tn)]:
    a = t[(t.model == m) & (t.alpha == .1) & (t.regime == "ID-all")]
    o[nm] = dict(random=float(a[a.scenario.isin(["DV-R", "BV-R"])].miss.mean()), shift=float(a[~a.scenario.isin(["DV-R", "BV-R"])].miss.mean()),
                 t100_miss=float(t[(t.model == m) & (t.alpha == .1) & (t.regime == "Target-100")].miss.mean()), pac_viol=float(t[(t.model == m) & (t.alpha == .1) & (t.regime == "TargetPAC-100")].viol2.mean()))
json.dump(o, open("results/cb_summary.json", "w"), indent=1); print(json.dumps(o, indent=1))

# ---- formatting audit table
fa = json.load(open("results/format_audit.json"))
rows = []
for ds, nm in (("BigVul", "Big-Vul"), ("DiverseVul", "DiverseVul")):
    d = fa[ds]; c = fa["main_lgbm_check"][ds]
    f = lambda k, s=100: f"{s * d[k]['vulnerable']:.1f} / {s * d[k]['benign']:.1f}"
    rows.append([nm, f"{d['auroc']:.3f}", f"{d['auprc_weighted']:.3f}", f("trailing_ws_lines"), f("starts_with_ws"), f("comment_markers", 1), f"{c['all']['auroc']:.3f} / {c['drop_chars_lines_tokens']['auroc']:.3f}"])
open("paper/tables/tab_format.tex", "w").write(tex(["Dataset", "AUROC", "AUPRC", "Trailing-ws lines (\\%)", "Starts with ws (\\%)", "Comment markers", "Main LGBM AUROC"], rows,
    "Formatting-shortcut audit on the stratified subsample (random 70/30 split). First two columns: LightGBM on 15 formatting features of the raw text only (indentation, trailing whitespace, comment markers, line lengths; AUPRC weighted to true prevalence). Next three: mean per function, vulnerable / benign. Last: AUROC of the main-study LightGBM with all metrics / with size-sensitive metrics (chars, lines, tokens) removed (seed 0).", "tab:format", "lcccccc"))

# ---- matched-seed table for the fine-tuned model
if HAS_FT_ANY:
    f = dn[dn.model == "ft_cb"]; keys = set(zip(f.scenario, f.seed))
    sel = lambda d: d[[(s, sd) in keys for s, sd in zip(d.scenario, d.seed)]]
    dm, tm = sel(dn), sel(tn)
    MODS = [("tf_ens", "TF-IDF+LGBM"), ("cb_ens", "CB frozen"), ("ft_cb", "CB fine-tuned"), ("ft_tf_ens", "TF+CB-FT")]
    rows = []
    for s in ORDER:
        ns = len({sd for sc_, sd in keys if sc_ == s})
        r = [SCN[s], str(ns)]
        r += [f"{dm[(dm.model == m) & (dm.scenario == s)].auroc.mean():.3f}" for m, _ in MODS]
        r += [f"{dm[(dm.model == m) & (dm.scenario == s)].auprc.mean():.3f}" for m, _ in MODS]
        r += [f"{100 * tm[(tm.model == m) & (tm.scenario == s) & (tm.regime == 'ID-all') & (tm.alpha == .1)].miss.mean():.1f}" for m, _ in MODS]
        rows.append(r)
    hdr = ["Scenario", "Seeds"] + [f"AUROC {n}" for _, n in MODS] + [f"AUPRC {n}" for _, n in MODS] + [f"miss {n}" for _, n in MODS]
    open("paper/tables/tab_ft.tex", "w").write(tex(hdr, rows, "Fine-tuned CodeBERT on normalised text compared with the other scorers on exactly the same (scenario, seed) pairs (seed 0 for all scenarios, seed 1 additionally for BV-R, BV-P, BV-T and DV-R; the other models in Tables~\\ref{tab:cbdetect} and~\\ref{tab:cbtriage} use three seeds). AUROC, AUPRC (weighted to true prevalence) and realised miss rate (\\%) of source-calibrated clearance at $\\alpha{=}10\\%$.", "tab:ft", "l" + "c" * 14))
    a = tm[(tm.alpha == .1) & (tm.regime == "ID-all")]; o = {}
    for m, _ in MODS:
        x = a[a.model == m]; o[m] = dict(random=float(x[x.scenario.isin(["DV-R", "BV-R"])].miss.mean()), shift=float(x[~x.scenario.isin(["DV-R", "BV-R"])].miss.mean()))
    for m in ("tf_ens", "cb_ens", "ft_cb"):
        b = tm[(tm.alpha == .1) & (tm.model == m)]
        o[m]["t100_miss"] = [float(b[(b.regime == "Target-100")].miss.min()), float(b[b.regime == "Target-100"].miss.max())]
    k = tm[(tm.regime == "ID-all") & tm.ks.notna() & (tm.model == "ft_cb")]; o["ks"] = [int(((k.miss - k.alpha) > k.ks).sum()), len(k)]
    json.dump(o, open("results/ft_summary.json", "w"), indent=1); print(json.dumps(o, indent=1))
