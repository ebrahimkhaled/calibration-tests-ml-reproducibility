"""Recompute the true calibration error of every Study B rule with the tie-safe 50-bin measure.

Each replicate is rebuilt from its own seed. The test-part predictions are compared with the stored file
(they must be identical, so the stored test results stay valid), and the new meta file is written to
results/simB_meta2/. Usage:  python 03b_true_error_v2.py [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module  # noqa: E402

sim = import_module("03_sim_classifiers")
from common import MODELS, RECALS, fit_recal, make_model, safe_auc, true_calibration_error  # noqa: E402

OUT = sim.ROOT / "results" / "simB_meta2"


def one(task):
    n_tot, rate, rep = task
    stem = f"N{n_tot}_r{int(rate * 100):02d}_rep{rep:03d}"
    f = OUT / f"{stem}_meta.csv"
    if f.exists():
        return 0.0
    stored = pd.read_csv(sim.OUT / f"{stem}.csv.gz")
    seed = int(n_tot * 7 + rate * 1000) * 1000 + rep
    rng = np.random.default_rng(seed)
    a = sim.TM["intercepts"][str(rate)]
    idx = rng.choice(sim.DRAW_IDX, n_tot, replace=False)
    X, p = sim.XV[idx], expit(a + sim.ETA0[idx])
    y = rng.binomial(1, p)
    i_tr, i_ca, i_te = np.split(np.arange(n_tot), [n_tot // 2, 3 * n_tot // 4])
    Xp, pp = sim.XV[sim.POOL_IDX], expit(a + sim.ETA0[sim.POOL_IDX])
    assert np.array_equal(stored.y.values, y[i_te]), stem
    rows = [dict(model="truth", recal="none", auc=safe_auc(y[i_te], p[i_te]), ce_true=0.0, mae_true=0.0,
                 brier=float(np.mean((y[i_te] - p[i_te]) ** 2)), maxdiff=0.0)]
    for m in MODELS:
        fit = make_model(m, seed).fit(X[i_tr], y[i_tr])
        raw_ca, raw_te = fit.predict_proba(X[i_ca])[:, 1], fit.predict_proba(X[i_te])[:, 1]
        raw_pool = fit.predict_proba(Xp)[:, 1]
        for r in RECALS:
            g = fit_recal(r, raw_ca, y[i_ca])
            q, qp = g(raw_te), g(raw_pool)
            rows.append(dict(model=m, recal=r, auc=safe_auc(y[i_te], q), ce_true=true_calibration_error(qp, pp),
                             mae_true=float(np.mean(np.abs(qp - pp))), brier=float(np.mean((y[i_te] - q) ** 2)),
                             maxdiff=float(np.max(np.abs(q - stored[f"{m}_{r}"].values)))))
    pd.DataFrame(rows).assign(N=n_tot, rate=rate, rep=rep, n_test=len(i_te),
                              events_test=int(y[i_te].sum())).to_csv(f, index=False)
    return time.time()


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = sorted([(n, r, k) for k in range(200) for n in sim.SIZES for r in sim.RATES], key=lambda t: -t[0])
    if len(sys.argv) > 2 and sys.argv[2] == "reverse":      # a second process can work from the other end
        tasks = tasks[::-1]
    t0 = time.time()
    with Pool(workers) as pool:
        list(pool.imap_unordered(one, tasks))
    d = pd.concat(pd.read_csv(f) for f in OUT.glob("*_meta.csv"))
    print("finished in", round(time.time() - t0), "s | largest difference from stored predictions:",
          d.maxdiff.max())
