#negaliu pakesti R kalbos. Tragedija. Jau geriau viska daryt sausu SQL

args <- commandArgs(trailingOnly = TRUE)
input_file <- if (length(args) >= 1) args[1] else "A21.csv"
output_file <- if (length(args) >= 2) args[2] else "A21_without_outliers.csv"
report_file <- if (length(args) >= 3) args[3] else "A21_outlier_report.csv"
correlation_file <- if (length(args) >= 4) args[4] else "A21_feature_correlations.csv"
iqr_multiplier <- 1.5

if (!file.exists(input_file)) {
  stop(paste("Input file does not exist:", input_file))
}

data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA", "NaN")
)

required_columns <- c(
  "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
  "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
  "Extent", "Solidity", "roundness", "Compactness", "ShapeFactor1",
  "ShapeFactor2", "ShapeFactor3", "ShapeFactor4", "class"
)

missing_columns <- c()
for (column_name in required_columns) {
  if (!(column_name %in% names(data))) {
    missing_columns <- c(missing_columns, column_name)
  }
}
if (length(missing_columns) > 0) {
  stop(paste("Missing required columns:", paste(missing_columns, collapse = ", ")))
}

numeric_columns <- c()
for (column_name in names(data)) {
  if (column_name != "class") {
    numeric_columns <- c(numeric_columns, column_name)
  }
}

row_count <- nrow(data)
column_count <- length(numeric_columns)
bounded_columns <- c("Eccentricity", "Extent", "Solidity", "roundness", "Compactness")

# -----------------------------------Skaitiniu stulpeliu sutvarkymas----------------------------------------

clean_number_text <- function(text) {
  cleaned <- gsub("[^0-9eE+.-]", "", text)
  if (cleaned == "") {
    return(NA_character_)
  }
  return(cleaned)
}

invalid_value_counts <- list()
for (column_name in numeric_columns) {
  invalid_value_counts[[column_name]] <- 0
}

for (column_name in numeric_columns) {
  original_column <- data[[column_name]]
  cleaned_column <- rep(NA_real_, row_count)

  for (row_index in seq_len(row_count)) {
    original_value <- original_column[row_index]

    if (is.na(original_value)) {
      next
    }

    numeric_value <- suppressWarnings(as.numeric(clean_number_text(original_value)))

    if (is.na(numeric_value)) {
      invalid_value_counts[[column_name]] <- invalid_value_counts[[column_name]] + 1
    }

    cleaned_column[row_index] <- numeric_value
  }

  data[[column_name]] <- cleaned_column
}

total_invalid_values <- 0
for (column_name in numeric_columns) {
  total_invalid_values <- total_invalid_values + invalid_value_counts[[column_name]]
}

# Kiekvieno skaitinio stulpelio reikšmės, pasiimtos vieną kartą, kad nereikėtų kaskart iš naujo skaityti.
column_values <- list()
for (column_name in numeric_columns) {
  column_values[[column_name]] <- data[[column_name]]
}

# ----------------------------------Patikrinimas loginiu duomenu rysiu---------------------------------------

logical_violation <- list()
for (column_name in numeric_columns) {
  logical_violation[[column_name]] <- rep(FALSE, row_count)
}

for (row_index in seq_len(row_count)) {
  for (column_name in numeric_columns) {
    value <- column_values[[column_name]][row_index]

    if (is.na(value)) {
      next
    }

    violation <- value <= 0

    if (column_name %in% bounded_columns && value > 1) {
      violation <- TRUE
    }
    if (column_name == "AspectRation" && value < 1) {
      violation <- TRUE
    }

    if (violation) {
      logical_violation[[column_name]][row_index] <- TRUE
    }
  }

  # Geometriniai ryšiai, kurie turi galioti tarp išvestinių matavimų.
  area <- column_values[["Area"]][row_index]
  convex_area <- column_values[["ConvexArea"]][row_index]
  if (!is.na(area) && !is.na(convex_area) && convex_area < area) {
    logical_violation[["ConvexArea"]][row_index] <- TRUE
  }

  major_axis <- column_values[["MajorAxisLength"]][row_index]
  minor_axis <- column_values[["MinorAxisLength"]][row_index]
  if (!is.na(major_axis) && !is.na(minor_axis) && major_axis < minor_axis) {
    logical_violation[["MajorAxisLength"]][row_index] <- TRUE
  }
}

logical_violation_rows <- rep(FALSE, row_count)
for (row_index in seq_len(row_count)) {
  for (column_name in numeric_columns) {
    if (logical_violation[[column_name]][row_index]) {
      logical_violation_rows[row_index] <- TRUE
      break
    }
  }
}

# ------------------------------------Isskirtys pagal IQR---------------------------------------

outlier_flag <- list()
for (column_name in numeric_columns) {
  outlier_flag[[column_name]] <- rep(FALSE, row_count)
}

lower_fence_by_column <- list()
upper_fence_by_column <- list()
outlier_count_by_column <- list()
logical_violation_count_by_column <- list()

for (column_name in numeric_columns) {
  values <- column_values[[column_name]]
  violations <- logical_violation[[column_name]]

  reference_values <- c()
  for (row_index in seq_len(row_count)) {
    if (!is.na(values[row_index]) && !violations[row_index]) {
      reference_values <- c(reference_values, values[row_index])
    }
  }

  lower_fence <- NA_real_
  upper_fence <- NA_real_
  outlier_count <- 0

  if (length(reference_values) >= 2) {
    quartiles <- quantile(reference_values, probs = c(0.25, 0.75), names = FALSE)
    iqr_value <- quartiles[2] - quartiles[1]
    lower_fence <- quartiles[1] - iqr_multiplier * iqr_value
    upper_fence <- quartiles[2] + iqr_multiplier * iqr_value

    for (row_index in seq_len(row_count)) {
      value <- values[row_index]
      if (!is.na(value) && (value < lower_fence || value > upper_fence)) {
        outlier_flag[[column_name]][row_index] <- TRUE
        outlier_count <- outlier_count + 1
      }
    }
  }

  lower_fence_by_column[[column_name]] <- lower_fence
  upper_fence_by_column[[column_name]] <- upper_fence
  outlier_count_by_column[[column_name]] <- outlier_count
  logical_violation_count_by_column[[column_name]] <- sum(violations)
}

statistical_outlier_rows <- rep(FALSE, row_count)
for (row_index in seq_len(row_count)) {
  for (column_name in numeric_columns) {
    if (outlier_flag[[column_name]][row_index]) {
      statistical_outlier_rows[row_index] <- TRUE
      break
    }
  }
}

# ---------------------------------Loginiu salygu pazeidimu ir isskirciu salynimas------------------------------------------
duplicated_rows <- duplicated(data)

rows_to_remove <- rep(FALSE, row_count)
for (row_index in seq_len(row_count)) {
  rows_to_remove[row_index] <- logical_violation_rows[row_index] || statistical_outlier_rows[row_index] || duplicated_rows[row_index]
}

cleaned_data <- data[!rows_to_remove, , drop = FALSE]

# Patikrinama, ar nelieka loginiu pazeidimu
for (row_index in seq_len(row_count)) {
  if (!rows_to_remove[row_index] && logical_violation_rows[row_index]) {
    stop("Cleaning failed: logical boundary violations remain.")
  }
}

write.csv(cleaned_data, output_file, row.names = FALSE, na = "")

# ----------------------------------Ataskaitos---------------------------------------

fence_report <- data.frame(
  column = numeric_columns,
  lower_fence = rep(NA_real_, column_count),
  upper_fence = rep(NA_real_, column_count),
  outlier_values = rep(0L, column_count),
  logical_boundary_violations = rep(0L, column_count),
  stringsAsFactors = FALSE
)

for (column_index in seq_len(column_count)) {
  column_name <- numeric_columns[column_index]
  fence_report$lower_fence[column_index] <- lower_fence_by_column[[column_name]]
  fence_report$upper_fence[column_index] <- upper_fence_by_column[[column_name]]
  fence_report$outlier_values[column_index] <- outlier_count_by_column[[column_name]]
  fence_report$logical_boundary_violations[column_index] <- logical_violation_count_by_column[[column_name]]
}

write.csv(fence_report, report_file, row.names = FALSE)

missing_value_count <- 0
for (column_name in names(data)) {
  missing_value_count <- missing_value_count + sum(is.na(data[[column_name]]))
}

summary_file <- sub("\\.csv$", "_summary.csv", report_file)
summary_report <- data.frame(
  metric = c(
    "rows_before", "logical_boundary_rows_removed",
    "statistical_outlier_rows_removed", "rows_removed_total", "rows_after",
    "invalid_values_converted_to_NA", "missing_values_left_untouched",
    "duplicate_rows_removed",
    "duplicate_rows_left_untouched"
  ),
  value = c(
    row_count, sum(logical_violation_rows), sum(statistical_outlier_rows),
    sum(rows_to_remove), nrow(cleaned_data),
    total_invalid_values, missing_value_count,
    duplicate_rows_removed <- sum(duplicate_rows),
    sum(duplicated(data))
  ),
  stringsAsFactors = FALSE
)
write.csv(summary_report, summary_file, row.names = FALSE)

if (row_count >= 2) {
  feature_correlations <- cor(data[numeric_columns], use = "pairwise.complete.obs")
} else {
  feature_correlations <- matrix(
    NA_real_, nrow = column_count, ncol = column_count,
    dimnames = list(numeric_columns, numeric_columns)
  )
}
write.csv(feature_correlations, correlation_file, row.names = TRUE)

cat(sprintf("Rows before: %d\n", row_count))
cat(sprintf("Logical-boundary rows removed: %d\n", sum(logical_violation_rows)))
cat(sprintf("Statistical-outlier rows removed: %d\n", sum(statistical_outlier_rows)))
cat(sprintf("Rows after: %d\n", nrow(cleaned_data)))
cat(sprintf("Cleaned data: %s\n", output_file))
cat(sprintf("Outlier report: %s\n", report_file))
cat(sprintf("Summary report: %s\n", summary_file))
cat(sprintf("Feature correlations: %s\n", correlation_file))
