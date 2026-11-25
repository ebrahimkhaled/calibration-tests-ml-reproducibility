"""Write the two paper tables (LaTeX-style rows read by the Word builder) straight from the result files,
and print every number the text quotes, so no number is typed by hand."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "manuscript" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.0f}"


B = pd.read_csv(RES / "studyB_long.csv.gz")

# ---------------------------------------------------------------- Table I: Study B, one cell
N, RATE = 16000, 0.15
cell = B[(B.N == N) & (B.rate == RATE)]
one = cell.drop_duplicates(["rep", "column"])
avg = one.groupby(["model", "recal"])[["auc", "ce_pts"]].mean()
rej = cell.pivot_table(index=["model", "recal"], columns="test", values="reject", aggfunc="mean")
order = [("truth", "none")] + [(m, r) for m in ["LR", "RF", "XGB", "NN"] for r in ["none", "platt", "isotonic", "temperature"]]
lines = []
for m, r in order:
    rname = {"none": "raw", "platt": "Platt", "isotonic": "isotonic", "temperature": "temperature"}[r]
    name = "True risk" if m == "truth" else f"{m}, {rname}"
    a = avg.loc[(m, r)]
    vals = [f"{a.auc:.3f}", f"{a.ce_pts:.1f}"] + [pct(rej.loc[(m, r), t]) for t in ["HL", "GiViTI", "Cox", "EDGE", "ECE"]]
    lines.append(" & ".join([name] + vals) + r" \\")
(TAB / "studyB.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nTable I (N = {N}, rate = {RATE}, reps = {one.rep.nunique()})")
print("\n".join(lines))

# ---------------------------------------------------------------- Table II: real data, raw and Platt
R = pd.read_csv(RES / "real_summary.csv")
long = pd.read_csv(RES / "real_long.csv.gz")
med_slope = long[long.test == "HL"].groupby(["data", "model", "recal"]).slope.median()
computed = long.groupby(["data", "column", "test"]).p.apply(lambda s: s.notna().mean()).to_dict()
for k, v in sorted(computed.items()):
    if v < 1:
        print("not always computed:", k, f"{v:.2f}")
nice = {"pima": "Pima", "breast_cancer": "Breast", "heart": "Heart", "cdc": "CDC"}
lines = []
for dname in ["heart", "breast_cancer", "pima", "cdc"]:
    for m in ["LR", "RF", "XGB", "NN"]:
        s = R[(R.data == dname) & (R.model == m)].set_index("recal")
        raw, pl = s.loc["none"], s.loc["platt"]
        vals = [f"{raw.auc:.2f}", f"{med_slope[(dname, m, 'none')]:.2f}", f"{100 * raw.ece:.1f}"]
        def cell(t, recal, rate):              # "n/a" when the test could not be computed in most splits
            ok = computed.get((dname, f"{m}_{recal}", t), 1.0)
            return "n/a" if ok < 0.5 else pct(rate)

        for t in ["HL", "Cox", "EDGE", "ECE"]:
            a, b = cell(t, "none", raw[t]), cell(t, "platt", pl[t])
            vals.append("n/a" if a == b == "n/a" else f"{a}/{b}")
        lines.append(" & ".join([f"{nice[dname]}, {m}"] + vals) + r" \\")
(TAB / "real.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nTable II")
print("\n".join(lines))
