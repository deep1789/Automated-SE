#!/bin/sh
# Reproduces every number, table and figure of the paper (CPU only; ~1.5 h on 4 cores).
set -e
pip install -q -r requirements.txt
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
python3 src/make_tables.py && python3 src/make_figs.py && python3 src/make_figs2.py
python3 tests_triage.py
(cd paper && latexmk -pdf main.tex)
