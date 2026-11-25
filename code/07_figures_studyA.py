"""Study A: rejection rates, the size table and the power heat map."""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
files = glob.glob(str(ROOT / "results" / "simA" / "*.csv"))
d = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
d["reject"] = (d.p < 0.05).astype(float).where(d.p.notna())      # missing p = not computed, left out of the rate
rates = (d.groupby(["rate", "n", "family", "target", "test"])
          .agg(power=("reject", "mean"), reps=("reject", "count"), na=("p", lambda s: s.isna().mean()))
          .reset_index())
rates["mcse"] = np.sqrt(rates.power * (1 - rates.power) / rates.reps)
rates.to_csv(ROOT / "results" / "studyA_rates.csv", index=False)
print("replicates per cell:", rates.reps.min(), "-", rates.reps.max(), "| cells:", len(rates) // 9,
      "| missing p:", round(rates.na.max(), 3))

# size table: test x (n, rate)
size = rates[rates.family == "none"].pivot_table(index="test", columns=["n", "rate"], values="power")
size = size.reindex(fs.TEST_ORDER)
size.to_csv(ROOT / "results" / "studyA_size.csv")
print(size.round(3).to_string())

# power heat map: rows = tests, columns = error level, panels = family x n, at a 15% event rate
fs.setup()
cmap = LinearSegmentedColormap.from_list("pw", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
fams = ["shift", "overconfident", "underconfident", "local"]
fam_label = {"shift": "Shift (all risks too high)", "overconfident": "Overconfident (too extreme)",
             "underconfident": "Underconfident (too timid)", "local": "Local (high-risk band only)"}
sizes = [250, 4000]
RATE = 0.15
fig, axes = plt.subplots(len(sizes), 4, figsize=(fs.FULL, 3.75), layout="constrained", sharey=True)
for i, n in enumerate(sizes):
    for j, fam in enumerate(fams):
        ax = axes[i, j]
        sub = rates[(rates.rate == RATE) & (rates.n == n) & (rates.family.isin([fam, "none"]))]
        m = sub.pivot_table(index="test", columns="target", values="power").reindex(fs.TEST_ORDER)
        # vector cells (pcolormesh), so the heat map stays sharp in the PDF
        im = ax.pcolormesh(np.arange(m.shape[1] + 1) - 0.5, np.arange(m.shape[0] + 1) - 0.5, m.values,
                           cmap=cmap, vmin=0, vmax=1)
        ax.set_ylim(m.shape[0] - 0.5, -0.5)
        for (r, c), v in np.ndenumerate(m.values):
            if np.isfinite(v):
                lab = "1" if v >= 1 else ">.99" if v >= 0.995 else f"{v:.2f}".lstrip("0")
                ax.text(c, r, lab, ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black")
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels([f"{t * 100:g}" for t in m.columns])
        ax.set_yticks(range(len(fs.TEST_ORDER)))
        ax.set_yticklabels([fs.TEST_NAME[t] for t in fs.TEST_ORDER])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title(fam_label[fam], fontsize=8)
        if j == 3:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(f"n = {n}", rotation=270, labelpad=10)
cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01, aspect=30)
cb.set_label("Rejection rate at the 5% level")
cb.ax.tick_params(labelsize=7.5)
fig.supxlabel("True calibration error (points)", fontsize=8)
fs.save(fig, "fig_power_heatmap")
