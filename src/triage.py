"""Risk-controlled triage: conformal auto-pass, weighted/Mondrian variants, certified auto-flag, metrics."""
import numpy as np
from scipy.stats import beta as _beta
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------- conformal auto-pass
def pvalues_pos(cal_pos, s):
    """Conformal p-value that a test point is 'vulnerable-like': p=(1+#{cal_pos<=s})/(n+1).
    Under exchangeability with the calibration positives, P(p<=alpha | y=1) <= alpha."""
    cal_pos = np.sort(np.asarray(cal_pos))
    k = np.searchsorted(cal_pos, np.asarray(s), side="right")
    return (1.0 + k) / (len(cal_pos) + 1.0)

def pvalues_pos_w(cal_pos_scores, w_cal, s, w_test):
    """Weighted conformal p-value under covariate shift (Tibshirani et al., 2019)."""
    o = np.argsort(cal_pos_scores)
    sc = np.asarray(cal_pos_scores)[o]; w = np.asarray(w_cal)[o]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    k = np.searchsorted(sc, s, side="right")
    num = cw[k] + w_test
    return num / (cw[-1] + w_test)

def auto_pass_mask(cal_pos, s, alpha, **kw):
    return pvalues_pos(cal_pos, s) <= alpha

def threshold_for_alpha(cal_pos, alpha):
    """Largest tau such that 'auto-pass iff s<=tau' coincides with the conformal rule p(s)=(1+#{cal_pos<=s})/(n+1) <= alpha."""
    sp_ = np.sort(cal_pos); n = len(sp_)
    kmax = int(np.floor(alpha * (n + 1) + 1e-12)) - 1      # pass iff #{cal_pos <= s} <= kmax  <=>  s < sp_[kmax]
    if kmax < 0: return -np.inf
    if kmax >= n: return np.inf
    return np.nextafter(sp_[kmax], -np.inf)


_PAC = {}
def pac_index(n, alpha, delta):
    """Largest j such that P(Beta(j, n+1-j) > alpha) <= delta  (continuous scores: miss rate of 'pass iff s < S_(j)' is Beta(j, n+1-j))."""
    key = (n, round(alpha, 6), delta)
    if key not in _PAC:
        j = np.arange(1, n + 1)
        ok = _beta.ppf(1 - delta, j, n + 1 - j) <= alpha
        _PAC[key] = int(j[ok].max()) if ok.any() else 0
    return _PAC[key]

def threshold_pac(cal_pos, alpha, delta=0.1):
    """High-probability variant: P(miss rate of the calibrated rule > alpha) <= delta over the draw of the calibration set."""
    sp_ = np.sort(cal_pos); n = len(sp_); j = pac_index(n, alpha, delta)
    return -np.inf if j == 0 else np.nextafter(sp_[j - 1], -np.inf)

# ---------------------------------------------------------------- certified auto-flag (fixed-sequence, Clopper-Pearson)
def cp_lower(k, n, delta):
    return 0.0 if k == 0 else float(_beta.ppf(delta, k, n - k + 1))

def flag_threshold(cal_scores, cal_y, prec_target, delta=0.05, grid=200, seed=0):
    """Learn-then-Test fixed-sequence procedure for auto-flagging.
    Half A of the calibration data fixes a descending grid of thresholds; half B runs the sequential Clopper-Pearson tests
    H_lam: precision(lam) < prec_target. Returns the smallest certified threshold (inf if none): P(precision(lam_hat) < target) <= delta."""
    rng = np.random.RandomState(seed); perm = rng.permutation(len(cal_scores)); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    qs = np.quantile(cal_scores[A], np.linspace(0.999, 0.5, grid))
    sB, yB = cal_scores[B], cal_y[B]; best = np.inf
    for lam in qs:
        m = sB >= lam; n = int(m.sum())
        if n < 10: continue
        if cp_lower(int(yB[m].sum()), n, delta) >= prec_target: best = lam
        else: break
    return best

# ---------------------------------------------------------------- Mondrian (bucketed) auto-pass
def mondrian_pass(cal_pos, cal_pos_bucket, s, bucket, alpha):
    out = np.zeros(len(s), bool)
    for b in np.unique(bucket):
        cb = cal_pos[cal_pos_bucket == b]
        if len(cb) == 0: continue
        m = bucket == b
        out[m] = pvalues_pos(cb, s[m]) <= alpha
    return out

# ---------------------------------------------------------------- metrics
def ece(p, y, bins=15):
    o = np.argsort(p); p, y = p[o], y[o]
    e = 0.0
    for idx in np.array_split(np.arange(len(p)), bins):
        if len(idx): e += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(e)

def to_prob(cal_s, cal_y, s, method):
    if method == "raw": return s
    if method == "platt":
        lr = LogisticRegression(C=1e6, max_iter=500).fit(cal_s.reshape(-1, 1), cal_y); return lr.predict_proba(s.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal_s, cal_y).predict(s)
    raise ValueError

def detection_metrics(s, y, thr):
    pred = s >= thr
    return dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), f1=f1_score(y, pred),
                mcc=matthews_corrcoef(y, pred), prec=float((y[pred] == 1).mean()) if pred.any() else 0.0,
                rec=float(pred[y == 1].mean()))

def best_f1_threshold(s, y):
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 200)))
    f = [f1_score(y, s >= t) for t in qs]
    return qs[int(np.argmax(f))]

def triage_stats(s, y, passed):
    """passed: boolean auto-pass mask. Returns miss rate (frac of vulnerable fns auto-passed) and clearance rate."""
    pos = y == 1
    return dict(miss=float(passed[pos].mean()) if pos.any() else np.nan, clear=float(passed.mean()),
                missed_n=int(passed[pos].sum()), n_pos=int(pos.sum()),
                npv=float((y[passed] == 0).mean()) if passed.any() else np.nan)
