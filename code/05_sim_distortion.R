# Study A: known risk p, predictions q = a known distortion of p, so the true calibration error is exact.
#   logit p ~ Normal(mu, 1.5^2), mu set for the event rate
#   shift:          logit q = logit p + d
#   overconfident:  logit q = c + s logit p,  s > 1, c keeps mean(q) = mean(p)
#   underconfident: the same with s < 1
#   local:          logit q = logit p + g exp(-(logit p - h)^2 / (2 * 0.5^2)), h = 90th percentile
# Strength is chosen so that the true calibration error E|q - p| equals 0, 0.5, 1, 2 or 4 points.
# Chunks of 50 replicates are written to disk as they finish; finished chunks are skipped.
# Usage: Rscript 05_sim_distortion.R [workers]
args <- commandArgs(trailingOnly = TRUE)
workers <- if (length(args)) as.integer(args[1]) else 16L
here <- normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
out <- file.path(here, "..", "results", "simA")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

SIGMA <- 1.5
RATES <- c(0.05, 0.15, 0.35)
SIZES <- c(250, 1000, 4000)
TARGETS <- c(0.005, 0.01, 0.02, 0.04)
FAMILIES <- c("shift", "overconfident", "underconfident", "local")
CHUNK <- 50L

set.seed(20270323)
z <- rnorm(4e5)
mu_for <- function(r) uniroot(function(m) mean(plogis(m + SIGMA * z)) - r, c(-10, 5))$root

distort <- function(eta, fam, k, ref) {
  switch(fam,
    shift = eta + k,
    overconfident = , underconfident = {
      cc <- uniroot(function(c) mean(plogis(c + k * ref)) - mean(plogis(ref)), c(-10, 10))$root
      cc + k * eta
    },
    local = eta + k * exp(-(eta - quantile(ref, 0.9))^2 / (2 * 0.25)))
}

# strength k that gives the target error, found on the large reference sample
params <- list()
for (r in RATES) {
  ref <- mu_for(r) + SIGMA * z
  p <- plogis(ref)
  for (fam in FAMILIES) for (tg in TARGETS) {
    err <- function(k) mean(abs(plogis(distort(ref, fam, k, ref)) - p)) - tg
    rng <- switch(fam, shift = c(0, 3), overconfident = c(1, 4), underconfident = c(0.02, 1), local = c(0, 6))
    k <- uniroot(err, rng)$root
    params[[length(params) + 1]] <- data.frame(rate = r, family = fam, target = tg, k = k)
  }
}
params <- do.call(rbind, params)
write.csv(params, file.path(out, "..", "simA_parameters.csv"), row.names = FALSE)
print(params)

cells <- rbind(expand.grid(rate = RATES, n = SIZES, family = "none", target = 0, chunk = 1:20,
                           stringsAsFactors = FALSE),
               expand.grid(rate = RATES, n = SIZES, family = FAMILIES, target = TARGETS, chunk = 1:10,
                           stringsAsFactors = FALSE))
cat(nrow(cells), "chunks of", CHUNK, "replicates\n")

run_chunk <- function(i) {
  cl <- cells[i, ]
  f <- file.path(out, sprintf("r%02d_n%d_%s_t%03d_c%02d.csv", round(cl$rate * 100), cl$n, cl$family,
                              round(cl$target * 1000), cl$chunk))
  if (file.exists(f)) return(0)
  mu <- mu_for(cl$rate)
  ref <- mu + SIGMA * z
  k <- if (cl$family == "none") NA else params$k[params$rate == cl$rate & params$family == cl$family &
                                                    abs(params$target - cl$target) < 1e-9]
  set.seed(i * 100003L)
  res <- vector("list", CHUNK)
  for (j in seq_len(CHUNK)) {
    eta <- mu + SIGMA * rnorm(cl$n)
    p <- plogis(eta)
    y <- rbinom(cl$n, 1, p)
    q <- if (cl$family == "none") p else plogis(distort(eta, cl$family, k, ref))
    q <- pmin(pmax(q, 1e-3), 1 - 1e-3)
    b <- battery(y, q, B = 500)
    res[[j]] <- cbind(rep = (cl$chunk - 1) * CHUNK + j, b[, c("test", "p")])
  }
  d <- do.call(rbind, res)
  write.csv(cbind(rate = cl$rate, n = cl$n, family = cl$family, target = cl$target, d), f, row.names = FALSE)
  1
}

library(parallel)
cl <- makeCluster(workers)
clusterExport(cl, c("cells", "params", "z", "SIGMA", "CHUNK", "out", "here", "mu_for", "distort", "run_chunk"))
invisible(clusterEvalQ(cl, {
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")
  source(file.path(here, "tests_lib.R"))
}))
t0 <- Sys.time()
ord <- order(-cells$n)                       # big jobs first
if (length(args) >= 2 && args[2] == "reverse") ord <- rev(ord)   # a second process can start from the other end
done <- parLapplyLB(cl, ord, run_chunk)
stopCluster(cl)
cat(sprintf("finished %d chunks (%d new) in %.1f min\n", nrow(cells), sum(unlist(done)),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
