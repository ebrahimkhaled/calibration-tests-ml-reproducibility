"""Real data: reliability diagrams (CDC data) and how often each test rejects on each data set."""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
from common import safe_auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

meta = pd.concat((pd.read_csv(f) for f in glob.glob(str(RES / "real" / "*_meta.csv"))), ignore_index=True)
meta["column"] = meta.model + "_" + meta.recal
tests = pd.concat((pd.read_csv(f) for f in glob.glob(str(RES / "real_tests" / "*_tests.csv"))), ignore_index=True)
k = tests.file.str.extract(r"^(.*)_split(\d+)")
tests["data"], tests["split"] = k[0], k[1].astype(int)
d = tests.merge(meta, on=["data", "split", "column"], how="left")
# a test that could not be computed (p missing) is left out of the rate, not counted as "not rejected"
d["reject"] = (d.p < 0.05).astype(float).where(d.p.notna())
d.to_csv(RES / "real_long.csv.gz", index=False)
print("splits:", d.drop_duplicates(["data", "split"]).groupby("data").size().to_dict())

summ = (d.pivot_table(index=["data", "model", "recal"], columns="test", values="reject", aggfunc="mean")
        .join(d.drop_duplicates(["data", "split", "column"]).groupby(["data", "model", "recal"])
              [["auc", "ece", "slope", "oe", "n_test"]].mean()))
summ.to_csv(RES / "real_summary.csv")
pd.set_option("display.width", 250)
print(summ[["n_test", "auc", "ece", "slope", "oe", "HL", "Spiegelhalter", "Cox", "CITL", "EDGE", "ECE"]].round(3).to_string())

# ---------------------------------------------------------------- reliability diagrams
# Pima: test parts of all 100 splits pooled (the procedure, not one lucky split); CDC: split 0 (63,420 patients).
fs.setup()


def pooled(name, splits):
    return pd.concat(pd.read_csv(RES / "real" / f"{name}_split{s:03d}.csv.gz") for s in splits)


PANELS = [("pima", "XGB", "Pima, XGB", range(100), 0.9),
          ("pima", "NN", "Pima, NN", range(100), 0.9),
          ("cdc", "LR", "CDC, LR", [0], 0.62),
          ("cdc", "RF", "CDC, RF", [0], 0.62)]
fig, axes = plt.subplots(2, 2, figsize=(fs.COL, 3.15), layout="constrained")
axes = axes.ravel()
cache = {}
for ax, (name, m, title, splits, lim) in zip(axes, PANELS):
    if name not in cache:
        cache[name] = pooled(name, splits)
    y = cache[name]
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.6, ls=":")
    for r, col, mk in [("none", fs.MODEL_COLOR[m], "o"), ("platt", "#444444", "s")]:
        q = y[f"{m}_{r}"].values
        groups = np.array_split(np.argsort(q, kind="stable"), 15)
        mq = np.array([q[g].mean() for g in groups])
        my = np.array([y.y.values[g].mean() for g in groups])
        nn = np.array([len(g) for g in groups])
        se = np.sqrt(np.clip(my * (1 - my), 1e-9, None) / nn)
        ax.fill_between(mq, my - 1.96 * se, my + 1.96 * se, color=col, alpha=0.18, lw=0)
        ax.plot(mq, my, color=col, marker=mk, ms=2.0, lw=0.9 if r == "none" else 0.7,
                ls="-" if r == "none" else "--", label="raw" if r == "none" else "after Platt")
    row = summ.loc[(name, m, "none")]
    sl = d[(d.data == name) & (d.column == f"{m}_none") & (d.test == "HL")].slope.median()

    def ece15(q):                       # ECE of the predictions drawn in this panel (15 equal-width bins)
        b = np.minimum((q * 15).astype(int), 14)
        return np.abs(np.bincount(b, weights=y.y.values - q, minlength=15)).sum() / len(q)

    ax.set_title(title, fontsize=8, pad=2)
    note = [f"AUC {row.auc:.2f}, slope {sl:.2f}",
            f"ECE {100 * ece15(y[f'{m}_none'].values):.1f} → {100 * ece15(y[f'{m}_platt'].values):.1f}"]
    ax.text(0.03, 0.97, chr(10).join(note), transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
for ax in axes[2:]:
    ax.set_xlabel("Predicted risk")
for ax in axes[::2]:
    ax.set_ylabel("Observed share")
axes[1].legend(loc="lower right", fontsize=7.5, handlelength=1.8)
fs.save(fig, "fig_reliability_real")
