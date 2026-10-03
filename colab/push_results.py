#!/usr/bin/env python3
"""Push fine-tuning score files from Colab to a separate branch of the GitHub repo (so they can be fetched without Drive/zip).
Usage (Colab):  python colab/push_results.py --src $OUT --repo https://github.com/<user>/<repo>.git --branch colab-results
The token is read from the environment variable GITHUB_TOKEN (set it from Colab Secrets; never hard-code or commit it).
Only *.parquet and *_done.json files are copied, into colab_results/ on the target branch. Safe to re-run: existing files are updated, nothing is deleted."""
import argparse, glob, os, shutil, subprocess, sys, tempfile

def run(cmd, cwd=None, secret=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout)
        if secret: msg = msg.replace(secret, "***")
        sys.exit(f"command failed: {' '.join(c.replace(secret, '***') if secret else c for c in cmd)}\n{msg}")
    return r.stdout

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--repo", required=True)
ap.add_argument("--branch", default="colab-results"); ap.add_argument("--base", default=None, help="branch to start from if the target branch does not exist yet")
ap.add_argument("--subdir", default="colab_results"); ap.add_argument("--allow-no-token", action="store_true", help="for tests with a local remote")
a = ap.parse_args()
tok = os.environ.get("GITHUB_TOKEN", "")
if not tok and not a.allow_no_token: sys.exit("GITHUB_TOKEN is not set (Colab: add it under the key icon > Secrets, then expose it as an environment variable).")
url = a.repo.replace("https://", f"https://x-access-token:{tok}@") if tok and a.repo.startswith("https://") else a.repo
files = sorted(glob.glob(os.path.join(a.src, "*.parquet")) + glob.glob(os.path.join(a.src, "*_done.json")))
if not files: sys.exit(f"no result files in {a.src}")
tmp = tempfile.mkdtemp()
try:
    run(["git", "clone", "-q", "--depth", "1", "--branch", a.branch, url, tmp], secret=tok)
except SystemExit:
    # target branch does not exist yet: create it from the base (or default) branch
    shutil.rmtree(tmp, ignore_errors=True); tmp = tempfile.mkdtemp()
    run(["git", "clone", "-q", "--depth", "1"] + (["--branch", a.base] if a.base else []) + [url, tmp], secret=tok)
    run(["git", "checkout", "-q", "-b", a.branch], cwd=tmp)
dst = os.path.join(tmp, a.subdir); os.makedirs(dst, exist_ok=True)
for f in files: shutil.copy(f, dst)
run(["git", "add", a.subdir], cwd=tmp)
if not run(["git", "status", "--porcelain"], cwd=tmp).strip(): print("nothing new to push"); sys.exit(0)
run(["git", "-c", "user.name=colab", "-c", "user.email=colab@example.invalid", "commit", "-q", "-m", f"Add Colab fine-tuning scores ({len(files)} files)"], cwd=tmp)
run(["git", "push", "-q", "origin", f"HEAD:{a.branch}"], cwd=tmp, secret=tok)
print(f"pushed {len(files)} files to branch '{a.branch}' under {a.subdir}/")
