# R packages used by the tests (R 4.4). The paper used ebrahim.gof 3.1.0 and givitiR 1.3 from CRAN.
install.packages(c("ebrahim.gof", "givitiR"), repos = "https://cloud.r-project.org")
stopifnot(packageVersion("ebrahim.gof") >= "3.1.0")
