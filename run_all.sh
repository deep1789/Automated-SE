#!/bin/sh
# Reproduces every number, table and figure of the paper (CPU only; ~1.5 h on 4 cores).
set -e
pip install -q -r requirements.txt
pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q transformers
python3 - <<'PY'
from huggingface_hub import snapshot_download
for r in ["claudios/DiverseVul", "bstee615/bigvul"]:
    snapshot_download(r, repo_type="dataset", local_dir=f"data/raw/{r.split('/')[1]}")
PY
python3 src/prep.py                    # normalise, deduplicate, audit -> data/clean.parquet, results/audit.json
python3 src/feats.py                   # hashed TF-IDF features + code metrics
mkdir -p results/cache
for s in 0 1 2; do for sc in BV-R BV-P BV-T DV-R DV-P DV2BV BV2DV; do echo "$sc $s"; done; done > /tmp/jobs.txt
export OMP_NUM_THREADS=2 NJ=2         # avoid OpenMP oversubscription
cat /tmp/jobs.txt | xargs -P 2 -L 1 sh -c 'python3 src/run_models.py $0 $1 lr,svm,lgbm'
python3 src/analyze.py                 # detection / calibration / triage / flag / bucket / CWE / cost tables (CSV)
for sc in DV-R DV-P BV-R BV-P BV-T DV2BV BV2DV; do FJ=2 python3 src/flaw_baseline.py $sc; done
python3 src/robust.py DV-R BV-R
python3 src/explain.py
# --- RQ6: frozen CodeBERT on a stratified subsample (raw and normalised text), formatting-shortcut audit
python3 src/cb_embed.py raw            # ~45 min CPU
python3 src/cb_embed.py norm           # ~45 min CPU
(for s in 0 1 2; do for sc in BV-R BV-P BV-T DV-R DV-P DV2BV BV2DV; do echo "$sc $s"; done; done) > /tmp/jobs_cb.txt
EMB=raw  xargs -P 2 -L 1 sh -c 'python3 src/run_cb.py $0 $1' < /tmp/jobs_cb.txt
EMB=norm xargs -P 2 -L 1 sh -c 'python3 src/run_cb.py $0 $1' < /tmp/jobs_cb.txt
python3 src/analyze_cb.py raw && python3 src/analyze_cb.py norm
PYTHONPATH=src python3 src/format_audit.py
# --- third corpus (PrimeVul), online control, shortcut ablation
python3 - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["primevul_train.jsonl", "primevul_valid.jsonl", "primevul_test.jsonl"]:
    hf_hub_download("colin/PrimeVul", f, repo_type="dataset", local_dir="data/raw/PrimeVul")
PY
PYTHONPATH=src python3 src/prep_pv.py && PYTHONPATH=src python3 src/feats_pv.py
(for s in 0 1 2; do for sc in PV-R PV-P PV-T DV2PV PV2DV PV2BV BV2PV; do echo "$sc $s"; done; done) | xargs -P 2 -L 1 sh -c 'python3 src/run_models.py $0 $1 lr,svm,lgbm'
AN_SCEN=PV-R,PV-P,PV-T,DV2PV,PV2DV,PV2BV,BV2PV AN_OUT=results/pv_ PYTHONPATH=src python3 src/analyze.py
PYTHONPATH=src python3 src/format_ablation.py && PYTHONPATH=src python3 src/format_audit.py
PYTHONPATH=src python3 - <<'PY'
# CVE order for the Big-Vul temporal stream
import glob, pandas as pd
from prep import norm, h
df = pd.concat([pd.read_parquet(f, columns=["CVE ID", "func_before"]) for f in sorted(glob.glob("data/raw/bigvul/data/*.parquet"))])
df["nh"] = [h(norm(c)) for c in df["func_before"]]
df["year"] = df["CVE ID"].str.extract(r"CVE-(\d{4})-")[0].astype(float); df["num"] = df["CVE ID"].str.extract(r"CVE-\d{4}-(\d+)")[0].astype(float)
df.drop_duplicates("nh")[["nh", "year", "num"]].to_parquet("data/bv_cve_order.parquet")
PY
PYTHONPATH=src python3 src/online.py
python3 src/make_tables.py && python3 src/make_figs.py && python3 src/make_figs2.py && python3 src/make_cb.py && python3 src/make_online.py && python3 src/make_pv.py
python3 tests_triage.py
(cd paper && latexmk -pdf main.tex)
