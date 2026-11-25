# Apply the test battery to every prediction file in a folder (resumes: finished files are skipped).
# Usage: Rscript 04_run_tests.R <input folder> <output folder> [workers] [B]
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; out <- args[2]
workers <- if (length(args) >= 3) as.integer(args[3]) else 8L
B <- if (length(args) >= 4) as.integer(args[4]) else 500L
dir.create(out, showWarnings = FALSE, recursive = TRUE)
files <- list.files(inp, pattern = "\\.csv\\.gz$", full.names = TRUE)
here <- normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
library(parallel)
cl <- makeCluster(workers)
clusterExport(cl, c("here", "out", "B"))
invisible(clusterEvalQ(cl, {
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")
  source(file.path(here, "tests_lib.R"))
}))
t0 <- Sys.time()
done <- parLapplyLB(cl, files, function(f) battery_file(f, out, B))
stopCluster(cl)
cat(sprintf("%d files (%d new) in %.1f min\n", length(files), sum(unlist(done)),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
