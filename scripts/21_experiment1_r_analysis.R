# Experiment 1 analysis in base R
#
# This script summarizes the current active repo state for:
# - Experiment 1a (Sinclair-style replication + jabberwocky extension)
# - Experiment 1b lexical-overlap version
# - Experiment 1b strict-control core version
# - lexical boost comparison between lexical-overlap and strict-control core 1b
#
# Notes:
# - The active repo currently contains summary/stats files for 1a, not the full
#   item-level score file, so the 1a section reads the stored paired statistics.
# - The 1b sections are fully rerunnable from item-level outputs that are present.

repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
exp1_root <- file.path(repo_root, "behavioral_results", "experiment-1")
output_root <- file.path(exp1_root, "r_analysis_outputs")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

set.seed(13)

bootstrap_mean_ci <- function(values, n_boot = 10000, conf = 0.95) {
  values <- as.numeric(values)
  alpha <- (1 - conf) / 2
  boot_means <- replicate(n_boot, mean(sample(values, length(values), replace = TRUE)))
  c(
    ci_low = as.numeric(stats::quantile(boot_means, alpha)),
    ci_high = as.numeric(stats::quantile(boot_means, 1 - alpha))
  )
}

sign_flip_p <- function(values, n_perm = 10000) {
  values <- as.numeric(values)
  observed <- mean(values)
  signs <- matrix(sample(c(-1, 1), length(values) * n_perm, replace = TRUE), nrow = n_perm)
  perm_means <- rowMeans(signs * rep(values, each = n_perm))
  mean(abs(perm_means) >= abs(observed))
}

exact_mcnemar_p <- function(b, c) {
  b <- as.integer(b)
  c <- as.integer(c)
  if ((b + c) == 0) {
    return(NA_real_)
  }
  stats::binom.test(b, b + c, p = 0.5, alternative = "two.sided")$p.value
}

paired_stats <- function(values) {
  values <- as.numeric(values)
  tt <- stats::t.test(values, mu = 0)
  sd_diff <- stats::sd(values)
  dz <- if (isTRUE(sd_diff > 0)) mean(values) / sd_diff else NA_real_
  ci <- bootstrap_mean_ci(values)
  data.frame(
    n_items = length(values),
    mean_diff = mean(values),
    sd_diff = sd_diff,
    effect_size_dz = dz,
    t_stat = unname(tt$statistic),
    t_p_two_sided = tt$p.value,
    perm_p_two_sided = sign_flip_p(values),
    bootstrap_ci95_low = ci[["ci_low"]],
    bootstrap_ci95_high = ci[["ci_high"]]
  )
}

read_csv_checked <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Missing file: %s", path))
  }
  read.csv(path, stringsAsFactors = FALSE)
}

load_1b_run <- function(run_dir) {
  list(
    summary = read_csv_checked(file.path(run_dir, "summary.csv")),
    stats = read_csv_checked(file.path(run_dir, "stats.csv")),
    items = read_csv_checked(file.path(run_dir, "item_scores.csv"))
  )
}

exp1a_report <- function() {
  summary_path <- file.path(exp1_root, "experiment-1a", "transitive_token_profiles", "transitive_item_summary.csv")
  paired_path <- file.path(exp1_root, "experiment-1a", "transitive_token_profiles", "stats", "paired_effects.csv")
  summary_df <- read_csv_checked(summary_path)
  paired_df <- read_csv_checked(paired_path)

  primary <- subset(summary_df, select = c(condition, target_structure, n_items, mean_sentence_pe_mean))
  wide <- reshape(
    primary,
    idvar = c("condition", "n_items"),
    timevar = "target_structure",
    direction = "wide"
  )
  names(wide) <- gsub("mean_sentence_pe_mean.", "", names(wide), fixed = TRUE)
  names(wide) <- gsub("n_items", "n_items", names(wide), fixed = TRUE)

  paired_primary <- subset(paired_df, metric == "sentence_pe_mean",
                           select = c(condition, mean_diff, bootstrap_ci95_low, bootstrap_ci95_high, t_p_two_sided, effect_size_dz))
  merged <- merge(wide, paired_primary, by = "condition")
  names(merged)[names(merged) == "active"] <- "active_pe_mean"
  names(merged)[names(merged) == "passive"] <- "passive_pe_mean"
  names(merged)[names(merged) == "mean_diff"] <- "passive_minus_active"
  merged
}

exp1b_overall <- function(run) {
  summary <- run$summary
  stats <- run$stats

  active_rate <- summary$passive_choice_rate[summary$prime_condition == "active"]
  passive_rate <- summary$passive_choice_rate[summary$prime_condition == "passive"]
  overall_row <- subset(stats, metric == "passive_choice_delta" & condition_a == "active" & condition_b == "passive")
  logprob_row <- subset(stats, metric == "logprob_delta" & condition_a == "active" & condition_b == "passive")

  data.frame(
    passive_choice_after_active = active_rate,
    passive_choice_after_passive = passive_rate,
    passive_choice_shift = overall_row$mean_diff_b_minus_a,
    passive_choice_ci_low = overall_row$bootstrap_ci95_low,
    passive_choice_ci_high = overall_row$bootstrap_ci95_high,
    passive_choice_p = overall_row$t_p_two_sided,
    logprob_shift = logprob_row$mean_diff_b_minus_a,
    logprob_ci_low = logprob_row$bootstrap_ci95_low,
    logprob_ci_high = logprob_row$bootstrap_ci95_high,
    logprob_p = logprob_row$t_p_two_sided
  )
}

exp1b_baseline_decomposition <- function(run) {
  summary <- run$summary
  stats <- run$stats
  baselines <- c("no_prime_eos", "no_prime_empty", "filler")

  rows <- lapply(baselines, function(baseline) {
    baseline_rate <- summary$passive_choice_rate[summary$prime_condition == baseline]

    active_row <- subset(stats, metric == "passive_choice_delta" & condition_a == "active" & condition_b == baseline)
    passive_row <- subset(stats, metric == "passive_choice_delta" & condition_a == "passive" & condition_b == baseline)

    active_priming <- active_row$mean_diff_b_minus_a
    passive_priming <- -passive_row$mean_diff_b_minus_a

    data.frame(
      baseline = baseline,
      baseline_passive_choice = baseline_rate,
      active_priming = active_priming,
      active_priming_ci_low = active_row$bootstrap_ci95_low,
      active_priming_ci_high = active_row$bootstrap_ci95_high,
      active_priming_p = active_row$t_p_two_sided,
      passive_priming = passive_priming,
      passive_priming_ci_low = -passive_row$bootstrap_ci95_high,
      passive_priming_ci_high = -passive_row$bootstrap_ci95_low,
      passive_priming_p = passive_row$t_p_two_sided,
      imbalance_passive_minus_active = passive_priming - active_priming
    )
  })

  do.call(rbind, rows)
}

lexical_boost_core <- function() {
  overlap_dir <- file.path(
    exp1_root, "experiment-1b", "processing_experiment_1b_gpt2large_v1_lexical-overlap", "processing_1b_core_core"
  )
  strict_dir <- file.path(
    exp1_root, "experiment-1b", "processing_experiment_1b_gpt2large_v3_strict-control", "processing_1b_core_core_lexically_controlled"
  )

  overlap_items <- read_csv_checked(file.path(overlap_dir, "item_scores.csv"))
  strict_items <- read_csv_checked(file.path(strict_dir, "item_scores.csv"))

  keep <- c("item_index", "prime_condition", "passive_choice_indicator", "passive_minus_active_logprob")
  merged <- merge(
    overlap_items[, keep],
    strict_items[, keep],
    by = c("item_index", "prime_condition"),
    suffixes = c("_overlap", "_strict")
  )

  per_condition <- do.call(
    rbind,
    lapply(split(merged, merged$prime_condition), function(df) {
      out_choice <- paired_stats(df$passive_choice_indicator_overlap - df$passive_choice_indicator_strict)
      out_choice$metric <- "passive_choice_indicator"
      out_choice$prime_condition <- df$prime_condition[1]
      out_choice$mean_overlap <- mean(df$passive_choice_indicator_overlap)
      out_choice$mean_strict <- mean(df$passive_choice_indicator_strict)

      out_logprob <- paired_stats(df$passive_minus_active_logprob_overlap - df$passive_minus_active_logprob_strict)
      out_logprob$metric <- "passive_minus_active_logprob"
      out_logprob$prime_condition <- df$prime_condition[1]
      out_logprob$mean_overlap <- mean(df$passive_minus_active_logprob_overlap)
      out_logprob$mean_strict <- mean(df$passive_minus_active_logprob_strict)

      rbind(out_choice, out_logprob)
    })
  )

  overlap_wide <- reshape(
    overlap_items[, c("item_index", "prime_condition", "passive_choice_indicator", "passive_minus_active_logprob")],
    idvar = "item_index",
    timevar = "prime_condition",
    direction = "wide"
  )
  strict_wide <- reshape(
    strict_items[, c("item_index", "prime_condition", "passive_choice_indicator", "passive_minus_active_logprob")],
    idvar = "item_index",
    timevar = "prime_condition",
    direction = "wide"
  )

  merged_wide <- merge(overlap_wide, strict_wide, by = "item_index", suffixes = c("_overlap", "_strict"))

  compare_delta <- function(prefix, cond_a, cond_b, label) {
    overlap_delta <- merged_wide[[sprintf("%s.%s_overlap", prefix, cond_b)]] - merged_wide[[sprintf("%s.%s_overlap", prefix, cond_a)]]
    strict_delta <- merged_wide[[sprintf("%s.%s_strict", prefix, cond_b)]] - merged_wide[[sprintf("%s.%s_strict", prefix, cond_a)]]
    diff <- overlap_delta - strict_delta
    out <- paired_stats(diff)
    out$metric <- prefix
    out$contrast <- label
    out$mean_overlap <- mean(overlap_delta)
    out$mean_strict <- mean(strict_delta)
    out
  }

  contrast_rows <- rbind(
    compare_delta("passive_choice_indicator", "active", "passive", "active_vs_passive_prime_contrast"),
    compare_delta("passive_minus_active_logprob", "active", "passive", "active_vs_passive_prime_contrast"),
    compare_delta("passive_choice_indicator", "active", "filler", "active_priming_vs_filler"),
    compare_delta("passive_minus_active_logprob", "active", "filler", "active_priming_vs_filler"),
    compare_delta("passive_choice_indicator", "filler", "passive", "passive_priming_vs_filler"),
    compare_delta("passive_minus_active_logprob", "filler", "passive", "passive_priming_vs_filler")
  )

  list(per_condition = per_condition, contrasts = contrast_rows)
}

write_table <- function(df, filename) {
  path <- file.path(output_root, filename)
  write.csv(df, path, row.names = FALSE)
  message("Wrote ", path)
}

main <- function() {
  exp1a <- exp1a_report()
  write_table(exp1a, "exp1a_primary_report.csv")

  overlap_core <- load_1b_run(file.path(
    exp1_root, "experiment-1b", "processing_experiment_1b_gpt2large_v1_lexical-overlap", "processing_1b_core_core"
  ))
  overlap_jab <- load_1b_run(file.path(
    exp1_root, "experiment-1b", "processing_experiment_1b_gpt2large_v1_lexical-overlap", "processing_1b_jabberwocky_jabberwocky"
  ))
  strict_core <- load_1b_run(file.path(
    exp1_root, "experiment-1b", "processing_experiment_1b_gpt2large_v3_strict-control", "processing_1b_core_core_lexically_controlled"
  ))

  write_table(exp1b_overall(overlap_core), "exp1b_v1_overlap_core_overall.csv")
  write_table(exp1b_baseline_decomposition(overlap_core), "exp1b_v1_overlap_core_baselines.csv")
  write_table(exp1b_overall(overlap_jab), "exp1b_v1_overlap_jabberwocky_overall.csv")
  write_table(exp1b_baseline_decomposition(overlap_jab), "exp1b_v1_overlap_jabberwocky_baselines.csv")
  write_table(exp1b_overall(strict_core), "exp1b_v3_strict_core_overall.csv")
  write_table(exp1b_baseline_decomposition(strict_core), "exp1b_v3_strict_core_baselines.csv")

  lexboost <- lexical_boost_core()
  write_table(lexboost$per_condition, "exp1b_core_lexical_boost_per_condition.csv")
  write_table(lexboost$contrasts, "exp1b_core_lexical_boost_contrasts.csv")

  message("\\nExperiment 1 analysis complete.")
}

main()
