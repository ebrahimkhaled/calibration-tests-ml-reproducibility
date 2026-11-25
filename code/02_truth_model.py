"""The true-risk model for the plasmode simulation.

The covariates are real BRFSS 2015 records (CDC Diabetes Health Indicators). The true risk is a
logistic model with curved and interaction terms, fitted once to all 253,680 records. Its curved part
is then made AMP times stronger, so that a main-effects logistic model is visibly wrong, as real
clinical relations often are. The intercept is moved to give each target event rate.

Writes results/truth_model.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
AMP = 2.0
RATES = [0.05, 0.15, 0.35]

cdc = pd.read_csv(ROOT / "data" / "clean" / "cdc.csv.gz")
X = cdc.drop(columns="y")
MEAN, SD = X.mean(), X.std()


def extra_terms(X):
    """Curved and interaction terms, on standardized columns."""
    s = (X - MEAN) / SD
    return pd.DataFrame({
        "BMI_sq": s.BMI ** 2,
        "BMI_cube": s.BMI ** 3,
        "Age_sq": s.Age ** 2,
        "GenHlth_sq": s.GenHlth ** 2,
        "HighBP_x_Age": s.HighBP * s.Age,
        "BMI_x_Sex": s.BMI * s.Sex,
        "HighBP_x_HighChol": s.HighBP * s.HighChol,
        "Age_x_GenHlth": s.Age * s.GenHlth,
    })


if __name__ == "__main__":
    S = (X - MEAN) / SD
    E = extra_terms(X)
    full = pd.concat([S, E], axis=1)
    m = LogisticRegression(C=np.inf, max_iter=5000).fit(full, cdc.y)
    b_main = m.coef_[0][: S.shape[1]]
    b_extra = AMP * m.coef_[0][S.shape[1]:]
    eta0 = S.values @ b_main + E.values @ b_extra
    intercepts = {str(r): brentq(lambda a: expit(a + eta0).mean() - r, -20, 20) for r in RATES}
    out = {"amp": AMP, "mean": MEAN.to_dict(), "sd": SD.to_dict(), "main": dict(zip(S.columns, b_main)),
           "extra": dict(zip(E.columns, b_extra)), "intercepts": intercepts,
           "fitted_intercept": float(m.intercept_[0])}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "truth_model.json").write_text(json.dumps(out, indent=1))
    for k, v in out["extra"].items():
        print(f"{k:18s} {v:+.3f}")
    print("intercepts", intercepts)
    p = expit(intercepts["0.15"] + eta0)
    print("true risk at 15%: quantiles", np.quantile(p, [0.01, 0.25, 0.5, 0.75, 0.99]).round(3))
