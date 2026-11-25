"""Study B: real covariates, known true risk, four classifiers, three recalibration maps.

One replicate:
  1. draw N real BRFSS records (not in the fixed 50,000-record reference pool),
  2. compute the true risk p from the truth model and draw y ~ Bernoulli(p),
  3. split 50 / 25 / 25 into training, recalibration and test parts,
  4. fit LR, RF, XGB and NN on the training part; fit each recalibration map on the second part,
  5. save y, p and the 17 prediction columns (truth + 4 x 4) on the test part, and
  6. record the true calibration error of every prediction rule on the reference pool.

Every replicate is written to its own file; finished files are skipped, so the run can resume.
Usage:  python 03_sim_classifiers.py [workers] [reps]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, RECALS, clip, fit_recal, make_model, safe_auc, true_calibration_error  # noqa: E402
from importlib import import_module  # noqa: E402

truth = import_module("02_truth_model")
from scipy.special import expit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "simB"
SIZES = [1000, 4000, 16000]          # total records; the test part is a quarter
RATES = [0.05, 0.15, 0.35]
POOL_N = 50_000

TM = json.loads((ROOT / "results" / "truth_model.json").read_text())
CDC = truth.cdc.drop(columns="y")
_S = (CDC - truth.MEAN) / truth.SD
ETA0 = _S.values @ np.array(list(TM["main"].values())) + truth.extra_terms(CDC).values @ np.array(list(TM["extra"].values()))
_perm = np.random.default_rng(2027).permutation(len(CDC))
POOL_IDX, DRAW_IDX = _perm[:POOL_N], _perm[POOL_N:]
XV = CDC.values.astype(float)


def one(task):
    n_tot, rate, rep = task
    f = OUT / f"N{n_tot}_r{int(rate * 100):02d}_rep{rep:03d}.csv.gz"
    if f.exists():
        return 0.0
    t0 = time.time()
    seed = int(n_tot * 7 + rate * 1000) * 1000 + rep
    rng = np.random.default_rng(seed)
    a = TM["intercepts"][str(rate)]
    idx = rng.choice(DRAW_IDX, n_tot, replace=False)
    X, p = XV[idx], expit(a + ETA0[idx])
    y = rng.binomial(1, p)
    i_tr, i_ca, i_te = np.split(np.arange(n_tot), [n_tot // 2, 3 * n_tot // 4])
    Xp, pp = XV[POOL_IDX], expit(a + ETA0[POOL_IDX])

    cols = {"y": y[i_te], "p": p[i_te], "truth": clip(p[i_te])}
    rows = [dict(model="truth", recal="none", auc=safe_auc(y[i_te], p[i_te]), ce_true=0.0,
                 mae_true=0.0, brier=float(np.mean((y[i_te] - p[i_te]) ** 2)))]
    for m in MODELS:
        if y[i_tr].sum() < 2:
            break
        fit = make_model(m, seed).fit(X[i_tr], y[i_tr])
        raw = {k: fit.predict_proba(X[i])[:, 1] for k, i in (("ca", i_ca), ("te", i_te))}
        raw_pool = fit.predict_proba(Xp)[:, 1]
        for r in RECALS:
            g = fit_recal(r, raw["ca"], y[i_ca])
            q, qp = g(raw["te"]), g(raw_pool)
            cols[f"{m}_{r}"] = q
            rows.append(dict(model=m, recal=r, auc=safe_auc(y[i_te], q),
                             ce_true=true_calibration_error(qp, pp), mae_true=float(np.mean(np.abs(qp - pp))),
                             brier=float(np.mean((y[i_te] - q) ** 2))))
    meta = pd.DataFrame(rows).assign(N=n_tot, rate=rate, rep=rep, n_test=len(i_te), events_test=int(y[i_te].sum()))
    meta.to_csv(f.with_name(f.name.replace(".csv.gz", "_meta.csv")), index=False)
    pd.DataFrame(cols).to_csv(f, index=False, float_format="%.6g")
    return time.time() - t0


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(n, r, k) for k in range(reps) for n in SIZES for r in RATES]
    tasks.sort(key=lambda t: -t[0])            # big jobs first, for better load balance
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for s in pool.imap_unordered(one, tasks):
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(tasks)} replicates, {time.time() - t0:.0f} s", flush=True)
    print("finished", len(tasks), "replicates in", round(time.time() - t0), "s")
