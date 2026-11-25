"""Study B: true error vs AUC, detection rate vs true error, and the recalibration table."""
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# true errors from the tie-safe 50-bin measure (03b_true_error_v2.py); the first-pass meta files are not used
meta = pd.concat((pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "simB_meta2" / "*_meta.csv"))),
                 ignore_index=True)
meta["column"] = np.where(meta.model == "truth", "truth", meta.model + "_" + meta.recal)
tests = pd.concat((pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "simB_tests" / "*_tests.csv"))),
                  ignore_index=True)
key = tests.file.str.extract(r"N(\d+)_r(\d+)_rep(\d+)").astype(int)
tests["N"], tests["rate"], tests["rep"] = key[0], key[1] / 100, key[2]
d = tests.merge(meta, on=["N", "rate", "rep", "column"], how="left")
# a test that could not be computed (p missing) is left out of the rate, not counted as "not rejected"
d["reject"] = (d.p < 0.05).astype(float).where(d.p.notna())
d["ce_pts"] = 100 * d.ce_true
print("rows", len(d), "| replicates", d.drop_duplicates(["N", "rate", "rep"]).shape[0],
      "| missing p", round(d.p.isna().mean(), 4), "| unmatched", d.ce_true.isna().sum())
d.to_csv(ROOT / "results" / "studyB_long.csv.gz", index=False)

# ---------------------------------------------------------------- table: by model and recalibration
TAB_TESTS = ["HL", "GiViTI", "Cox", "EDGE", "ECE"]
order = [("truth", "none")] + [(m, r) for m in ["LR", "RF", "XGB", "NN"] for r in ["none", "platt", "isotonic", "temperature"]]
tab = (d.pivot_table(index=["N", "rate", "model", "recal"], columns="test", values="reject", aggfunc="mean")
       .join(d.drop_duplicates(["N", "rate", "rep", "column"]).groupby(["N", "rate", "model", "recal"])
             [["auc", "ce_pts", "mae_true"]].mean()))
tab.to_csv(ROOT / "results" / "studyB_table_all.csv")
for N, rate in [(4000, 0.15), (16000, 0.15), (1000, 0.15), (4000, 0.05), (4000, 0.35)]:
    try:
        t = tab.loc[(N, rate)].reindex(order)[["auc", "ce_pts"] + TAB_TESTS]
    except KeyError:
        continue
    print(f"\nN = {N}, rate = {rate}\n", t.round(3).to_string())

# false alarms and misses against the true error
bins = [0, 0.5, 1, 1.5, 2, 3, 100]
labels = ["<0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", ">3"]
d["ce_bin"] = pd.cut(d.ce_pts, bins, labels=labels, right=False)
det = d.groupby(["N", "ce_bin", "test"], observed=True).agg(rate=("reject", "mean"), k=("reject", "count")).reset_index()
det.to_csv(ROOT / "results" / "studyB_detection.csv", index=False)
print("\n", det.pivot_table(index=["N", "test"], columns="ce_bin", values="rate", observed=True).round(2).to_string())
print("\n", det.pivot_table(index=["N", "test"], columns="ce_bin", values="k", observed=True).iloc[:3].to_string())

# ---------------------------------------------------------------- figure: AUC vs true calibration error
fs.setup()
one = d.drop_duplicates(["N", "rate", "rep", "column"])
sub = one[(one.N == 4000) & (one.rate == 0.15) & (one.model != "truth")]
LABEL_POS = {"LR": (0, 8, "center"), "RF": (12, -9, "left"), "XGB": (3, 8, "left"), "NN": (-3, 8, "right")}
fig, ax = plt.subplots(figsize=(fs.COL, 2.5), layout="constrained")
rng = np.random.default_rng(1)
for m in ["LR", "RF", "XGB", "NN"]:
    raw = sub[(sub.model == m) & (sub.recal == "none")]
    ax.scatter(raw.ce_pts, raw.auc, s=3, alpha=0.25, color=fs.MODEL_COLOR[m], linewidths=0, rasterized=True)
    pts = sub[sub.model == m].groupby("recal")[["ce_pts", "auc"]].median()
    x0, y0 = pts.loc["none"]
    ax.scatter([x0], [y0], s=26, color=fs.MODEL_COLOR[m], edgecolor="white", linewidth=0.6, zorder=4)
    dx, dy, ha = LABEL_POS[m]
    ax.annotate(m, (x0, y0), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8,
                color=fs.MODEL_COLOR[m], fontweight="bold")
    for r, mk in [("platt", "s"), ("isotonic", "^"), ("temperature", "D")]:
        x1, y1 = pts.loc[r]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=fs.MODEL_COLOR[m], mutation_scale=5,
                                    shrinkA=3, shrinkB=2))
        ax.scatter([x1], [y1], s=9, marker=mk, facecolor="white", edgecolor=fs.MODEL_COLOR[m], linewidth=0.7, zorder=5)
tr = one[(one.N == 4000) & (one.rate == 0.15) & (one.model == "truth")]
ax.axhline(tr.auc.median(), color="black", lw=0.6, ls=":")
ax.text(0.99, tr.auc.median() + 0.002, "AUC of the true risk", transform=ax.get_yaxis_transform(),
        fontsize=7.5, va="bottom", ha="right")
ax.set_xlabel("True calibration error (points)")
ax.set_ylabel("AUC on the test part")
from matplotlib.lines import Line2D  # noqa: E402
hs = [Line2D([], [], marker=mk, ls="", mfc="white", mec="black", mew=0.7, ms=3.5, label=fs.RECAL_NAME[r])
      for r, mk in [("platt", "s"), ("isotonic", "^"), ("temperature", "D")]]
ax.legend(handles=hs, loc="lower right", fontsize=7.5, handletextpad=0.1, ncol=3, columnspacing=0.8,
          title="After recalibration", title_fontsize=7.5)
fs.save(fig, "fig_auc_vs_error")

# ---------------------------------------------------------------- figure: detection rate vs true error
SHOW = ["HL", "GiViTI", "Cox", "EDGE", "ECE", "Spiegelhalter"]
style = {"HL": (fs.OI["blue"], "-", "o"), "Spiegelhalter": (fs.OI["orange"], "-", "s"), "GiViTI": (fs.OI["black"], ":", "x"),
         "Cox": (fs.OI["green"], "-", "^"), "EDGE": (fs.OI["vermillion"], "-", "D"), "ECE": (fs.OI["purple"], "--", "v")}
fig, axes = plt.subplots(1, 2, figsize=(fs.COL, 2.6), layout="constrained", sharey=True)
for ax, N in zip(axes, [1000, 16000]):
    for t in fs.TEST_ORDER:
        s = det[(det.N == N) & (det.test == t) & (det.k >= 100)]     # bins with at least 100 rules
        xs = [labels.index(b) for b in s.ce_bin.astype(str)]
        if t in SHOW:
            c, ls, mk = style[t]
            ax.plot(xs, s.rate, color=c, ls=ls, marker=mk, ms=3, label=fs.TEST_NAME[t], zorder=3)
        else:
            ax.plot(xs, s.rate, color="#BBBBBB", lw=0.6, zorder=1,
                    label="Other three tests" if t == "CITL" else None)
    ax.axhline(0.05, color="black", lw=0.5, ls=":")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7.5)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Test part of {N // 4} patients", fontsize=8)
    ax.set_xlabel("True calibration error (points)")
axes[0].set_ylabel("Share of rules flagged")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=7.5,
           handlelength=1.8, columnspacing=1.0)
fs.save(fig, "fig_detection")
