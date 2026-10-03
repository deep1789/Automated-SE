import numpy as np, sys
sys.path.insert(0, "src")
from triage import *
rng = np.random.RandomState(0)
for trial in range(200):
    n = rng.randint(5, 300); cal = np.round(rng.randn(n), rng.choice([1, 2, 6]))   # include ties
    s = np.round(rng.randn(500), 2); a = rng.choice([0.02, 0.05, 0.1, 0.2, 0.5])
    th = threshold_for_alpha(cal, a)
    assert ((s <= th) == (pvalues_pos(cal, s) <= a)).all(), (n, a)
# finite-sample validity: realised miss rate averaged over calibration draws is <= alpha (continuous scores, exchangeable)
for n in (20, 50, 200, 1000):
    for a in (0.05, 0.1, 0.2):
        ms = []
        for r in range(4000):
            cal = rng.randn(n); test = rng.randn(1)
            ms.append(test[0] <= threshold_for_alpha(cal, a))
        print(n, a, round(np.mean(ms), 4), "bound", a, "lower", round(np.floor(a*(n+1))/(n+1), 4))
        assert np.mean(ms) <= a + 3*np.sqrt(a*(1-a)/4000)
# weighted p-value with unit weights equals unweighted
cal = rng.randn(100); s = rng.randn(50)
assert np.allclose(pvalues_pos_w(cal, np.ones(100), s, np.ones(50)), pvalues_pos(cal, s))
print("ok")
