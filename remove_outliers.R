# Dry Bean logical-boundary and statistical-outlier analysis.
# Missing values and duplicate rows are intentionally left for a separate script.
# Usage:
#   Rscript remove_outliers.R
#   Rscript remove_outliers.R input.csv cleaned.csv report.csv correlations.csv

args <- commandArgs(trailingOnly = TRUE)
input_file <- if (length(args) >= 1) args[1] else "A21.csv"
output_file <- if (length(args) >= 2) args[2] else "A21_without_outliers.csv"
report_file <- if (length(args) >= 3) args[3] else "A21_outlier_report.csv"
correlation_file <- if (length(args) >= 4) args[4] else "A21_feature_correlations.csv"
iqr_multiplier <- 1.5

if (!file.exists(input_file)) {
  stop(sprintf("Input file does not exist: %s", input_file))
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
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_columns, collapse = ", ")))
}

numeric_columns <- setdiff(names(data), "class")
invalid_value_report <- data.frame(
  column = character(),
  invalid_values = integer(),
  stringsAsFactors = FALSE
)

# Normalize numeric fields, including values such as "1132.4 px".
for (column_name in numeric_columns) {
  original_values <- data[[column_name]]
  cleaned_values <- gsub("[^0-9eE+.-]", "", original_values)
  cleaned_values[cleaned_values == ""] <- NA_character_
  numeric_values <- suppressWarnings(as.numeric(cleaned_values))
  invalid_count <- sum(!is.na(original_values) & is.na(numeric_values))

  if (invalid_count > 0) {
    invalid_value_report <- rbind(
      invalid_value_report,
      data.frame(
        column = column_name,
        invalid_values = invalid_count,
        stringsAsFactors = FALSE
      )
    )
  }

  data[[column_name]] <- numeric_values
}

# Logical constraints for the documented Dry Bean measurements.
logical_violation_flags <- matrix(
  FALSE,
  nrow = nrow(data),
  ncol = length(numeric_columns),
  dimnames = list(NULL, numeric_columns)
)

for (column_name in numeric_columns) {
  values <- data[[column_name]]
  invalid <- !is.na(values) & values <= 0

  if (column_name %in% c("Eccentricity", "Extent", "Solidity", "roundness", "Compactness")) {
    invalid <- invalid | (!is.na(values) & values > 1)
  }
  if (column_name == "AspectRation") {
    invalid <- invalid | (!is.na(values) & values < 1)
  }

  logical_violation_flags[, column_name] <- invalid
}

# Geometric relationships that must hold between derived measurements.
logical_violation_flags[, "ConvexArea"] <- logical_violation_flags[, "ConvexArea"] |
  (!is.na(data$Area) & !is.na(data$ConvexArea) & data$ConvexArea < data$Area)
logical_violation_flags[, "MajorAxisLength"] <- logical_violation_flags[, "MajorAxisLength"] |
  (!is.na(data$MajorAxisLength) & !is.na(data$MinorAxisLength) &
     data$MajorAxisLength < data$MinorAxisLength)

logical_violation_rows <- if (nrow(data) > 0) {
  apply(logical_violation_flags, 1, any)
} else {
  logical(0)
}

# Calculate IQR fences while ignoring missing values; missing rows are not removed.
outlier_flags <- matrix(
  FALSE,
  nrow = nrow(data),
  ncol = length(numeric_columns),
  dimnames = list(NULL, numeric_columns)
)
fence_report <- data.frame(
  column = numeric_columns,
  lower_fence = rep(NA_real_, length(numeric_columns)),
  upper_fence = rep(NA_real_, length(numeric_columns)),
  outlier_values = rep(0L, length(numeric_columns)),
  logical_boundary_violations = rep(0L, length(numeric_columns)),
  stringsAsFactors = FALSE
)

for (column_name in numeric_columns) {
  values <- data[[column_name]]
  reference_values <- values[!is.na(values) & !logical_violation_flags[, column_name]]

  if (length(reference_values) >= 2) {
    quartiles <- quantile(reference_values, probs = c(0.25, 0.75), names = FALSE)
    iqr_value <- quartiles[2] - quartiles[1]
    lower_fence <- quartiles[1] - iqr_multiplier * iqr_value
    upper_fence <- quartiles[2] + iqr_multiplier * iqr_value
    column_flags <- !is.na(values) & (values < lower_fence | values > upper_fence)

    outlier_flags[, column_name] <- column_flags
    fence_report[fence_report$column == column_name, c("lower_fence", "upper_fence")] <-
      c(lower_fence, upper_fence)
    fence_report[fence_report$column == column_name, "outlier_values"] <- sum(column_flags)
  }

  fence_report[fence_report$column == column_name, "logical_boundary_violations"] <-
    sum(logical_violation_flags[, column_name])
}

statistical_outlier_rows <- if (nrow(data) > 0) apply(outlier_flags, 1, any) else logical(0)
rows_to_remove <- logical_violation_rows | statistical_outlier_rows
cleaned_data <- data[!rows_to_remove, , drop = FALSE]

# Verify only the responsibilities of this script were applied.
if (nrow(cleaned_data) > 0 &&
    any(apply(logical_violation_flags[!rows_to_remove, , drop = FALSE], 1, any))) {
  stop("Cleaning failed: logical boundary violations remain.")
}

write.csv(cleaned_data, output_file, row.names = FALSE, na = "")

summary_file <- sub("\\.csv$", "_summary.csv", report_file)
summary_report <- data.frame(
  metric = c(
    "rows_before", "logical_boundary_rows_removed",
    "statistical_outlier_rows_removed", "rows_removed_total", "rows_after",
    "invalid_values_converted_to_NA", "missing_values_left_untouched",
    "duplicate_rows_left_untouched"
  ),
  value = c(
    nrow(data), sum(logical_violation_rows), sum(statistical_outlier_rows),
    sum(rows_to_remove), nrow(cleaned_data),
    sum(invalid_value_report$invalid_values), sum(is.na(data)),
    sum(duplicated(data))
  ),
  stringsAsFactors = FALSE
)
write.csv(summary_report, summary_file, row.names = FALSE)
write.csv(fence_report, report_file, row.names = FALSE)

if (nrow(data) >= 2) {
  feature_correlations <- cor(data[numeric_columns], use = "pairwise.complete.obs")
} else {
  feature_correlations <- matrix(
    NA_real_, nrow = length(numeric_columns), ncol = length(numeric_columns),
    dimnames = list(numeric_columns, numeric_columns)
  )
}
write.csv(feature_correlations, correlation_file, row.names = TRUE)

cat(sprintf("Rows before: %d\n", nrow(data)))
cat(sprintf("Logical-boundary rows removed: %d\n", sum(logical_violation_rows)))
cat(sprintf("Statistical-outlier rows removed: %d\n", sum(statistical_outlier_rows)))
cat(sprintf("Rows after: %d\n", nrow(cleaned_data)))
cat(sprintf("Cleaned data: %s\n", output_file))
cat(sprintf("Outlier report: %s\n", report_file))
cat(sprintf("Summary report: %s\n", summary_file))
cat(sprintf("Feature correlations: %s\n", correlation_file))
