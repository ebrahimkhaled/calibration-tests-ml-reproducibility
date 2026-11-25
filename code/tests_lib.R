# The calibration tests applied to one vector of outcomes y and predictions q.
# All tests except ECE come from ebrahim.gof::run.all.external (frozen predictions, external mode).
suppressPackageStartupMessages(library(ebrahim.gof))
# givitiR is only "Suggested" by ebrahim.gof; without it the belt column would silently be empty.
if (!requireNamespace("givitiR", quietly = TRUE)) stop("Please install givitiR (see install.R).")
if (packageVersion("ebrahim.gof") < "3.1.0") warning("The paper used ebrahim.gof 3.1.0; results may differ.")

# ebrahim.gof runs the GiViTI test in a fresh R subprocess (callr) to guard against a crash in givitiR's
# belt-drawing code. The test itself (givitiCalibrationTest) does not crash, and a subprocess per call made
# the simulation about 20 times slower, so the same call is made directly here. Same function, same p-value.
local({
  direct <- function(ctx, opts) {
    devel <- if (is.null(opts$devel)) "internal" else opts$devel
    pv <- tryCatch(givitiR::givitiCalibrationTest(o = as.numeric(ctx$y), e = as.numeric(ctx$ph),
                                                  devel = devel)$p.value, error = function(e) NA_real_)
    list(Statistic = NA_real_, df = NA_real_, p_value = if (length(pv) == 1) pv else NA_real_,
         Note = paste0("calibration belt; devel=", devel))
  }
  utils::assignInNamespace("gof_giviti", direct, "ebrahim.gof")
})

TESTS <- c("Hosmer-Lemeshow (external)" = "HL",
           "Spiegelhalter-z" = "Spiegelhalter",
           "Cox recalibration" = "Cox",
           "Calibration-in-the-large" = "CITL",
           "GiViTI" = "GiViTI",
           "Osius-Rojek (external)" = "OsiusRojek",
           "Stukel (offset)" = "Stukel",
           "EDGE (G=10)" = "EDGE")

# Expected calibration error with M equal-width bins (the usual machine-learning version).
ece <- function(y, q, M = 15) {
  b <- pmin(floor(q * M), M - 1)
  n <- length(y)
  sum(abs(rowsum(y - q, b))) / n          # sum_b (n_b / n) |mean y_b - mean q_b|
}

# Consistency-resampling test: under H0 the outcomes are Bernoulli(q), so draw them B times.
ece_test <- function(y, q, B = 500, M = 15) {
  e <- ece(y, q, M)
  b <- pmin(floor(q * M), M - 1)
  n <- length(y)
  Ystar <- matrix(stats::runif(n * B) < q, n, B)
  estar <- colSums(abs(rowsum(Ystar - q, b))) / n
  c(ece = e, p = (1 + sum(estar >= e)) / (B + 1))
}

# One row per test: p-value; plus the descriptive slope, O/E and ECE.
battery <- function(y, q, B = 500) {
  r <- suppressWarnings(run.all.external(y, q, G = 10))
  pv <- setNames(r$p_value[match(names(TESTS), r$Test)], TESTS)
  et <- ece_test(y, q, B)
  data.frame(test = c(names(pv), "ECE"), p = c(unname(pv), et[["p"]]),
             slope = r$Statistic[r$Test == "Calibration slope"],
             oe = r$Statistic[r$Test == "O/E ratio"], ece = et[["ece"]],
             stringsAsFactors = FALSE)
}

# Every prediction column of one file (all columns except y and p).
battery_file <- function(f, out_dir, B = 500) {
  out <- file.path(out_dir, sub("\\.csv\\.gz$", "_tests.csv", basename(f)))
  if (file.exists(out)) return(invisible(0))
  d <- utils::read.csv(f)
  # seed for the ECE resampling, taken from the file name (kept as run; files with the same character sum
  # share the resampling stream, which leaves each p-value valid)
  set.seed(sum(utf8ToInt(basename(f))) * 7919L)
  cols <- setdiff(names(d), c("y", "p"))
  res <- do.call(rbind, lapply(cols, function(cl) {
    b <- tryCatch(battery(d$y, d[[cl]], B), error = function(e)
      data.frame(test = NA, p = NA, slope = NA, oe = NA, ece = NA))
    cbind(column = cl, b)
  }))
  utils::write.csv(cbind(file = basename(f), res), out, row.names = FALSE)
  invisible(1)
}
