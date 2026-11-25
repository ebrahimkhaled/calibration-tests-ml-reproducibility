"""Shared pieces: the four classifiers, the three recalibration maps, and the error measures."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Setting penalty=None")
warnings.filterwarnings("ignore", category=FutureWarning)

EPS = 1e-3                      # predictions are kept inside [0.001, 0.999] before any test
MODELS = ["LR", "RF", "XGB", "NN"]
RECALS = ["none", "platt", "isotonic", "temperature"]


def make_model(name, seed):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                             tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
    if name == "NN":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=500,
                                           random_state=seed))
    raise ValueError(name)


def clip(q):
    return np.clip(q, EPS, 1 - EPS)


def fit_recal(kind, q_cal, y_cal):
    """Return a function that maps raw predictions to recalibrated ones."""
    z = logit(clip(q_cal))
    if kind == "none":
        return lambda q: clip(q)
    if kind == "platt":                       # q' = expit(a + b * logit q)
        m = LogisticRegression(C=np.inf, max_iter=1000).fit(z[:, None], y_cal)
        a, b = m.intercept_[0], m.coef_[0, 0]
        return lambda q: clip(expit(a + b * logit(clip(q))))
    if kind == "temperature":                 # q' = expit(logit q / T)

        def nll(log_t):
            s = expit(z / np.exp(log_t))
            s = np.clip(s, 1e-12, 1 - 1e-12)
            return -np.mean(y_cal * np.log(s) + (1 - y_cal) * np.log(1 - s))

        t = np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)
        return lambda q: clip(expit(logit(clip(q)) / t))
    if kind == "isotonic":                    # monotone step function (pool adjacent violators)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(q_cal, y_cal)
        return lambda q: clip(iso.predict(q))
    raise ValueError(kind)


def true_calibration_error(q, p, bins=50):
    """Calibration error when the true risk p is known: sum_b w_b |mean p_b - mean q_b| over bins of q
    with (about) equal counts. This is the expected gap between E[Y | q] and q, without outcome noise.
    Bin edges are quantiles of q and equal values always share a bin, so step-function predictions
    (isotonic regression, random forests) are not split arbitrarily."""
    edges = np.unique(np.quantile(q, np.linspace(0, 1, bins + 1)[1:-1]))
    b = np.searchsorted(edges, q, side="right")
    n_b = np.bincount(b)
    keep = n_b > 0
    gap = np.abs(np.bincount(b, weights=p)[keep] - np.bincount(b, weights=q)[keep]) / n_b[keep]
    return float(np.sum(n_b[keep] / len(q) * gap))


def safe_auc(y, q):
    return float(roc_auc_score(y, q)) if 0 < y.sum() < len(y) else np.nan
