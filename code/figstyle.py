"""One look for every figure: IEEE A4 column widths, Times at 7.5-8 pt, a colour-blind-safe palette."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
COL = 243.3 / 72          # one column of the PEIT A4 template, inches
FULL = 504.6 / 72         # both columns

# Okabe-Ito colours, safe for the common forms of colour blindness
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
MODEL_COLOR = {"LR": OI["blue"], "RF": OI["green"], "XGB": OI["vermillion"], "NN": OI["purple"], "truth": OI["black"]}
MODEL_NAME = {"LR": "Logistic regression", "RF": "Random forest", "XGB": "XGBoost", "NN": "Neural network",
              "truth": "True risk"}
RECAL_NAME = {"none": "Raw", "platt": "Platt", "isotonic": "Isotonic", "temperature": "Temperature"}
TEST_ORDER = ["HL", "Spiegelhalter", "Cox", "CITL", "GiViTI", "OsiusRojek", "Stukel", "EDGE", "ECE"]
TEST_NAME = {"HL": "Hosmer–Lemeshow", "Spiegelhalter": "Spiegelhalter’s z", "Cox": "Cox slope-and-intercept",
             "CITL": "Calibration-in-the-large", "GiViTI": "GiViTI belt", "OsiusRojek": "Osius–Rojek",
             "Stukel": "Stukel", "EDGE": "EDGE", "ECE": "ECE bootstrap"}


def setup():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
        "figure.dpi": 150, "axes.grid": False,
    })


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    # no tight bounding box: the saved width must equal the column width, so 8 pt stays 8 pt in the paper
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600)
    with mpl.rc_context({"svg.fonttype": "path"}):      # vector copy for Word; glyphs as outlines, so Word cannot re-space them
        fig.savefig(FIG_DIR / f"{name}.svg")
    plt.close(fig)
    print("saved", name)
