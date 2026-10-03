#!/usr/bin/env python3
"""Fine-tune CodeBERT on the 30k-function subsample for every (scenario, seed) of the study and save per-part scores.

Output (one parquet per scenario/seed/part, same row order as results/cbn_cache/*.parquet):
    {out}/{scenario}_s{seed}_{part}.parquet  with columns  nh, y, ft_cb  (ft_cb = logit(vulnerable) - logit(benign))
    {out}/{scenario}_s{seed}_done.json        training log (best epoch, validation AUROC, times)
The script is resumable: finished (scenario, seed) pairs are skipped, so a Colab disconnect only loses the current run.

Typical use (Colab, after `python colab/finetune_codebert.py --prepare`):
    python colab/finetune_codebert.py --seeds 0 --out /content/drive/MyDrive/ft_codebert
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd

SCENARIOS = ["BV-R", "BV-P", "BV-T", "DV-R", "DV-P", "DV2BV", "BV2DV"]

def prepare():
    """Download both corpora from the Hugging Face hub, run the study's pre-processing, and verify the universe checksum."""
    from huggingface_hub import snapshot_download
    for r in ["claudios/DiverseVul", "bstee615/bigvul"]:
        snapshot_download(r, repo_type="dataset", local_dir=f"data/raw/{r.split('/')[1]}")
    if not os.path.exists("data/clean.parquet"):
        prep_main()
    from cb_splits import write_meta, universe, checksum, DATASETS
    write_meta()
    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_checksum.json")))
    for ds in DATASETS:
        c = checksum(universe(ds))
        assert c == ref[ds], f"Universe mismatch for {ds}: got {c}, expected {ref[ds]} (dataset revision changed?)"
    print("data ready; universe checksums match the study's.")

def prep_main():
    import runpy; runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "prep.py"), run_name="__main__")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="download data + pre-process, then exit")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="ft_out"); ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-batch", type=int, default=128); ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline")
    ap.add_argument("--dummy", action="store_true", help="no model: random scores (checks splits/IO/merge only)")
    a = ap.parse_args()
    if a.prepare: return prepare()

    import torch
    from sklearn.metrics import roc_auc_score
    from cb_splits import universe, universe_texts, parts_for, DATASETS
    from run_models import SC
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not (a.smoke or a.dummy): sys.exit("No GPU found. In Colab: Runtime > Change runtime type > GPU.")
    amp_dtype = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"device={dev} amp={amp_dtype if dev=='cuda' else 'off'} model={a.model} max_len={a.max_len} epochs={a.epochs} batch={a.batch} lr={a.lr}", flush=True)

    U = {ds: universe(ds) for ds in DATASETS}
    tok = enc = None
    if not a.dummy:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        tok = AutoTokenizer.from_pretrained(a.model); enc = {}
        for ds in DATASETS:
            t0 = time.time(); txt = universe_texts(ds, U[ds])
            enc[ds] = tok(txt, truncation=True, max_length=(64 if a.smoke else a.max_len))["input_ids"]
            print(f"tokenised {ds}: {len(txt)} functions, mean length {np.mean([len(x) for x in enc[ds]]):.0f}, "
                  f"{np.mean([len(x) >= a.max_len for x in enc[ds]]):.0%} truncated ({time.time()-t0:.0f}s)", flush=True)

    def collate(ids_list):
        L = max(len(x) for x in ids_list); pad = tok.pad_token_id
        ids = torch.full((len(ids_list), L), pad, dtype=torch.long); att = torch.zeros((len(ids_list), L), dtype=torch.long)
        for i, x in enumerate(ids_list): ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
        return ids.to(dev), att.to(dev)

    @torch.no_grad()
    def predict(model, ids, idx):
        model.eval(); order = np.argsort([len(ids[i]) for i in idx]); out = np.zeros(len(idx), np.float32)
        for s in range(0, len(idx), a.eval_batch):
            sel = order[s:s + a.eval_batch]; x, m = collate([ids[idx[j]] for j in sel])
            with torch.autocast(dev, dtype=amp_dtype, enabled=(dev == "cuda")):
                lg = model(input_ids=x, attention_mask=m).logits.float()
            out[sel] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
        return out

    for seed in a.seeds:
        for sc in a.scenarios:
            tag = f"{sc}_s{seed}"; done = os.path.join(a.out, f"{tag}_done.json")
            if os.path.exists(done): print("skip (done)", tag); continue
            t0 = time.time(); S, T, kind = SC[sc]; mS, mT = U[S], U[T]
            parts = parts_for(sc, seed, mS, mT)
            tr = parts["train"][1]; ytr = mS.y.values[tr]
            rng = np.random.RandomState(seed); perm = rng.permutation(len(tr)); nv = int(a.val_frac * len(tr))
            if a.smoke: tr, ytr, nv = tr[:48], ytr[:48], 12; perm = rng.permutation(48)
            va, fit = tr[perm[:nv]], tr[perm[nv:]]   # validation (early stopping) is carved out of the training distribution; calibration data is never used for selection
            log = dict(scenario=sc, seed=seed, n_fit=int(len(fit)), n_val=int(len(va)), epochs=[])
            scores = {}
            if a.dummy:
                for k, (w, idx) in parts.items():
                    if k != "train": scores[k] = np.random.RandomState(seed).randn(len(idx)).astype(np.float32)
            else:
                torch.manual_seed(seed); np.random.seed(seed)
                model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)
                ids = enc[S]; y = mS.y.values
                steps = a.epochs * int(np.ceil(len(fit) / a.batch))
                opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
                sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
                scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda" and amp_dtype == torch.float16))
                best, best_state = -1, None
                for ep in range(a.epochs):
                    model.train(); order = np.random.RandomState(seed * 100 + ep).permutation(len(fit)); tl = 0; te = time.time()
                    for s in range(0, len(order), a.batch):
                        b = fit[order[s:s + a.batch]]; x, m = collate([ids[i] for i in b]); yb = torch.tensor(y[b], dtype=torch.long, device=dev)
                        with torch.autocast(dev, dtype=amp_dtype, enabled=(dev == "cuda")):
                            loss = torch.nn.functional.cross_entropy(model(input_ids=x, attention_mask=m).logits.float(), yb)
                        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item() * len(b)
                    vs = predict(model, ids, va); auc = roc_auc_score(y[va], vs) if len(set(y[va])) > 1 else float("nan")
                    log["epochs"].append(dict(epoch=ep + 1, train_loss=tl / len(fit), val_auroc=float(auc), seconds=time.time() - te))
                    print(f"[{tag}] epoch {ep+1}/{a.epochs} loss {tl/len(fit):.4f} val AUROC {auc:.4f} ({time.time()-te:.0f}s)", flush=True)
                    if not (auc <= best):   # also keeps the first epoch when AUROC is NaN
                        best, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(best_state); log["best_val_auroc"] = float(best)
                for k, (w, idx) in parts.items():
                    if k == "train": continue
                    if a.smoke: idx = idx[:32]
                    scores[k] = predict(model, enc[S if w == "S" else T], idx)
                del model, opt, best_state
                if dev == "cuda": torch.cuda.empty_cache()
            for k, (w, idx) in parts.items():
                if k == "train": continue
                m_ = mS if w == "S" else mT
                if a.smoke: idx = idx[:32]
                d = pd.DataFrame(dict(nh=m_.nh.iloc[idx].values, y=m_.y.values[idx], ft_cb=scores[k][:len(idx)])); d["part"] = k; d["scenario"] = sc; d["seed"] = seed
                d.to_parquet(os.path.join(a.out, f"{tag}_{k}.parquet"))
            log["seconds"] = time.time() - t0; json.dump(log, open(done, "w"))
            print(f"[{tag}] finished in {log['seconds']/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
