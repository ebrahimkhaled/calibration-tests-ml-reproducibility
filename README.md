# Reproduction materials: Accurate but Miscalibrated

Code, data and results for the paper **"Accurate but Miscalibrated: Testing the Risk Estimates of Medical
Machine Learning Classifiers"** by Ebrahim Khaled Ebrahim (Department of Applied Statistics, Faculty of
Business, Alexandria University; ORCID [0009-0006-7839-8778](https://orcid.org/0009-0006-7839-8778)).

The study asks whether statistical calibration tests detect miscalibrated medical classifiers, and how
recalibration (Platt, isotonic, temperature scaling) changes their verdicts. Nine tests are compared:
Hosmer–Lemeshow, Spiegelhalter's z, the Cox slope-and-intercept test, calibration-in-the-large, the GiViTI
calibration belt, Osius–Rojek, Stukel, EDGE and a consistency-resampling (bootstrap) test of the expected
calibration error (ECE).

- **Study A** — the true risk is known and the prediction is a known distortion of it (shift, over- and
  underconfidence, a local fault) with a true calibration error of 0 to 4 percentage points;
  n = 250 / 1000 / 4000; event rate 5 / 15 / 35%; 81,000 replicates.
- **Study B** — a plasmode design: real covariates of the CDC Diabetes Health Indicators data, simulated
  outcomes from a known logistic truth with curved terms; logistic regression, random forest, XGBoost and a
  neural network, each raw and after three recalibration maps; N = 1000 / 4000 / 16,000; 1800 replicates.
- **Real data** — Pima diabetes, Wisconsin breast cancer, Cleveland heart disease, CDC diabetes indicators.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | every script, in run order (below) |
| `data/raw/` | the four data sets as downloaded; `data/clean/` the cleaned versions (outcome column `y`) |
| `results/simA/` | Study A: one file per chunk of 50 replicates, with the p-value of every test |
| `results/simB_meta2/` | Study B: AUC and true calibration error of every prediction rule in every replicate |
| `results/simB_tests/` | Study B: p-values of every test for every rule and replicate |
| `results/real_tests/` | Real data: p-values, calibration slope, O/E ratio and ECE for every split and rule |
| `results/*.csv(.gz)` | summaries behind the figures and tables (`studyA_rates.csv`, `studyB_long.csv.gz`, ...) |
| `figures/` | the paper's figures (PDF, PNG at 600 dpi, SVG) |

The per-replicate *prediction* files of Study B (about 150 MB) and of the real-data splits (about 30 MB) are
not stored: scripts 03 and 06 rebuild them exactly from their seeds (checked: identical predictions).

## Software

Python 3.13 with the packages in `requirements.txt` (`pip install -r requirements.txt`) and R 4.4 with
`ebrahim.gof` 3.1.0 and `givitiR` 1.3 (`Rscript install.R`). `sessionInfo-R.txt` records the R session.
All tests except the ECE bootstrap come from `ebrahim.gof::run.all.external()`. `code/tests_lib.R` calls
`givitiR::givitiCalibrationTest()` directly for the calibration belt instead of through the package's
`callr` subprocess; the p-values are identical and the run is about 20 times faster.

## Run order

Run from the repository root. Worker counts are optional (default 8 or 16); times are for a 24-core PC.

```
python code/01_prepare_data.py                            # data/raw -> data/clean
python code/02_truth_model.py                             # Study B truth model -> results/truth_model.json
python code/03_sim_classifiers.py 16                      # Study B predictions -> results/simB/       (~15 min)
python code/03b_true_error_v2.py 16                       # true calibration error -> results/simB_meta2/ (~30 min)
Rscript code/04_run_tests.R results/simB results/simB_tests 16   # Study B tests          (~10 min)
python code/06_real_data.py 8                             # real-data splits -> results/real/            (~7 min)
Rscript code/04_run_tests.R results/real results/real_tests 16   # real-data tests        (~15 min)
Rscript code/05_sim_distortion.R 16                       # Study A -> results/simA/                     (~45-75 min)
python code/07_figures_studyA.py                          # Fig. 1
python code/08_figures_studyB.py                          # Figs. 2 and 3
python code/09_figures_real.py                            # Fig. 4
python code/10_tables.py                                  # Tables I and II (printed; also written to manuscript/tables)
```

Scripts 03, 03b, 04, 05 and 06 write every replicate to disk and skip finished files, so an interrupted
run resumes where it stopped. To redraw the figures and tables from the stored results, run 07 to 10 only.
Run the R scripts with `Rscript` (they locate their own folder from the command line, so `source()` from an
IDE will not work).

| Paper item | Script | Inputs |
|---|---|---|
| Fig. 1 (Study A heat map) | `07_figures_studyA.py` | `results/simA/` |
| Fig. 2 (AUC vs true error), Fig. 3 (detection) | `08_figures_studyB.py` | `results/simB_meta2/`, `results/simB_tests/` |
| Fig. 4 (reliability diagrams) | `09_figures_real.py` | `results/real_tests/`, `results/real/` (from 06) |
| Table I (Study B), Table II (real data) | `10_tables.py` | `results/studyB_long.csv.gz`, `results/real_summary.csv`, `results/real_long.csv.gz` |

## Seeds

Study A seeds each chunk of 50 replicates (`set.seed(i * 100003)`); Study B seeds each replicate from N, the
event rate and the replicate number; the real-data splits are seeded by data set and split. The resampling
of the ECE bootstrap in `04_run_tests.R` is seeded from the file name.

## Data sources and licences

| Data | Source | Licence |
|---|---|---|
| Pima Indians diabetes | OpenML data set 37 (originally Smith et al., 1988) | public (OpenML) |
| Wisconsin diagnostic breast cancer | UCI Machine Learning Repository, id 17, doi:10.24432/C5DW2B | CC BY 4.0 |
| Cleveland heart disease | UCI Machine Learning Repository, id 45, doi:10.24432/C52P4X | CC BY 4.0 |
| CDC Diabetes Health Indicators | UCI Machine Learning Repository, id 891, doi:10.24432/C53919, built from the CDC Behavioral Risk Factor Surveillance System 2015 | BRFSS data are a US Government work in the public domain; see the UCI page for the linked data set's terms |

Cleaning (`01_prepare_data.py`): Pima zeros in glucose, blood pressure, skin fold, insulin and BMI mean "not
measured" and are set to the overall median; for the heart data, any narrowing (`num > 0`) is the outcome
and the 6 rows with missing values are dropped.

## Licence and citation

Code: MIT (`LICENSE`). Results, figures and documentation: CC BY 4.0 (`LICENSE-DATA.txt`). The data sets keep
their own licences (table above). Please cite the paper and this archive (`CITATION.cff`).
