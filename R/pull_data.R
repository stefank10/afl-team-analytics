# Pull public AFL match results and player stats with fitzRoy, one CSV per season.
# Source: AFL Tables (afltables.com) via the fitzRoy package.
# Run from the repo root:  Rscript R/pull_data.R
#
# Why R for this step? fitzRoy is the best-maintained AFL data package and is R-only.
# Everything after the raw pull (cleaning, modelling, charts) is done in Python.

if (!requireNamespace("fitzRoy", quietly = TRUE)) {
  install.packages("fitzRoy", repos = "https://cloud.r-project.org")
}
library(fitzRoy)

seasons <- 2019:2026
out_dir <- file.path("data", "raw")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (s in seasons) {
  message("Season ", s, " ...")

  # Match results: one row per game (scores, venue, round, date)
  res <- tryCatch(fetch_results_afltables(season = s),
                  error = function(e) { message("  results failed: ", conditionMessage(e)); NULL })
  if (!is.null(res)) {
    write.csv(res, file.path(out_dir, paste0("results_", s, ".csv")), row.names = FALSE)
    message("  results: ", nrow(res), " rows")
  }

  # Player stats: one row per player per game (kicks, handballs, tackles, inside 50s, ...)
  ps <- tryCatch(fetch_player_stats_afltables(season = s),
                 error = function(e) { message("  player stats failed: ", conditionMessage(e)); NULL })
  if (!is.null(ps)) {
    write.csv(ps, file.path(out_dir, paste0("player_stats_", s, ".csv")), row.names = FALSE)
    message("  player stats: ", nrow(ps), " rows")
  }
}

message("Done. Files written to ", out_dir)
