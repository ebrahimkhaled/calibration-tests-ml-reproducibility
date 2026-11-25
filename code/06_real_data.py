"""Real data: repeated random splits of the four medical data sets.

Each split is stratified 50 / 25 / 25 into training, recalibration and test parts. The four classifiers
are fitted on the training part, each recalibration map on the second part, and the 16 prediction
columns on the test part are saved (one file per split; finished splits are skipped).
Usage:  python 06_real_data.py [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, RECALS, fit_recal, make_model, safe_auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real"
SPLITS = {"heart": 100, "breast_cancer": 100, "pima": 100, "cdc": 10}


def one(task):
    name, s = task
    f = OUT / f"{name}_split{s:03d}.csv.gz"
    if f.exists():
        return 0.0
    t0 = time.time()
    d = pd.read_csv(ROOT / "data" / "clean" / f"{name}.csv.gz")
    X, y = d.drop(columns="y").values.astype(float), d.y.values
    seed = 1000 * len(name) + s
    idx = np.arange(len(y))
    i_tr, i_rest = train_test_split(idx, train_size=0.5, stratify=y, random_state=seed)
    i_ca, i_te = train_test_split(i_rest, train_size=0.5, stratify=y[i_rest], random_state=seed)
    cols, rows = {"y": y[i_te]}, []
    for m in MODELS:
        fit = make_model(m, seed).fit(X[i_tr], y[i_tr])
        q_ca, q_te = fit.predict_proba(X[i_ca])[:, 1], fit.predict_proba(X[i_te])[:, 1]
        for r in RECALS:
            q = fit_recal(r, q_ca, y[i_ca])(q_te)
            cols[f"{m}_{r}"] = q
            rows.append(dict(model=m, recal=r, auc=safe_auc(y[i_te], q), brier=float(np.mean((y[i_te] - q) ** 2))))
    pd.DataFrame(rows).assign(data=name, split=s, n_test=len(i_te)).to_csv(
        f.with_name(f.name.replace(".csv.gz", "_meta.csv")), index=False)
    pd.DataFrame(cols).to_csv(f, index=False, float_format="%.6g")
    return time.time() - t0


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(n, s) for n, k in SPLITS.items() for s in range(k)]
    tasks.sort(key=lambda t: t[0] != "cdc")    # the large data set first
    t0 = time.time()
    with Pool(workers) as pool:
        list(pool.imap_unordered(one, tasks))
    print("finished", len(tasks), "splits in", round(time.time() - t0), "s")
