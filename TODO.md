# Ka dar reik mum padaryt

## 1. Fixes to existing code

### - [ ] 1.1 Outlier handling (HIGHEST PRIORITY)

- **Now:** IQR fences are computed on each of the 16 columns across the whole dataset. A row flagged in *any* column is deleted automatically.
- **Should be:** Outliers are *identified and investigated, not removed automatically* (slide 17). For each outlier we decide whether it's a measurement error, a rare bean, or a meaningful extreme value, and give a reason.
- **Change:**
  - Stop putting `statistical_outlier_rows` into `rows_to_remove`. Keep the flag only.
  - Compute fences **per class** as well as globally. Size features differ a lot between varieties (BOMBAY especially), so global fences will flag real beans.
  - Output `table(class, outlier_flag)` and the % of rows flagged. If one class is flagged almost entirely, that shows the global method is wrong and belongs in the report.
  - Count how many columns flag each row. Area / ConvexArea / Perimeter / EquivDiameter are near-duplicates, so one big bean gets counted 4–5 times.
  - Only remove rows that break physical rules (section 1.3). For statistical outliers, choose keep / cap / remove and justify it.

### - [ ] 1.2 Number parsing (`clean_number_text`)

- **Now:** Every character except digits, `e`, `+`, `-` and `.` is stripped silently. `"1.234,5"` → `1.2345`, `"12 34"` → `1234`. Only values that end up as NA are counted.
- **Should be:** Every changed value is counted and each type of corruption is handled on purpose.
- **Change:**
  - First list the non-numeric strings that actually appear in each column and handle each pattern explicitly (decimal comma, units, spaces, text).
  - Count **modified** values as well as values that became NA, and report both.
  - Anything that can't be parsed confidently becomes NA and goes to missing-value handling (2.1). Don't guess.

### - [ ] 1.3 Logical / physical validity checks

- **Now:** Checks for ≤ 0, bounded columns ≤ 1, AspectRation ≥ 1, ConvexArea ≥ Area, Major ≥ Minor. Good start.
- **Should be:** All physical constraints are checked, including the formula relationships between features.
- **Change:**
  - Add `ShapeFactor4 ≤ 1`.
  - Add identity checks, first measuring the tolerance on rows known to be clean:
    - `EquivDiameter ≈ sqrt(4 * Area / pi)`
    - `AspectRation ≈ MajorAxisLength / MinorAxisLength`
    - `Compactness ≈ EquivDiameter / MajorAxisLength`
    - `ShapeFactor3 ≈ Compactness^2`
    - `roundness ≈ 4 * pi * Area / Perimeter^2`
    - `ShapeFactor1 ≈ MajorAxisLength / Area`, `ShapeFactor2 ≈ MinorAxisLength / Area`
  - A row that breaks an identity has a corrupted value even if it looks statistically normal. Report these separately.

### - [ ] 1.4 Class label validation

- **Now:** The `class` column is never checked.
- **Should be:** Labels are validated and class balance is reported (slide 16).
- **Change:**
  - `table(data$class, useNA = "ifany")`. Look for typos, mixed case, stray spaces and missing labels.
  - Normalize labels (trim + upper case) and check that exactly 7 valid classes remain.
  - Report counts and proportions per class and comment on any imbalance.

### - [ ] 1.5 Duplicates

- **Now:** Exact duplicates are removed. `duplicate_rows_left_untouched` is always 0 because it's measured after removal.
- **Should be:** Duplicates are reported, removal is justified, and there are no meaningless metrics.
- **Change:**
  - Delete the `duplicate_rows_left_untouched` metric.
  - Change `duplicate_rows_removed <- sum(...)` inside `c()` into a normal assignment before the data frame.
  - Check duplicates **after** label normalization (`"seker"` vs `"SEKER"` hide duplicates) and decide whether to check with or without the class column.

### - [ ] 1.6 Dead / useless code

- **Now:** The final "Cleaning failed: logical boundary violations remain" loop. It can never fail, because `rows_to_remove` already contains every violation row.
- **Should be:** Checks that can actually fail.
- **Change:** Delete it, or replace it with a real post-cleaning check (e.g. re-run the validity checks on `cleaned_data` and `stopifnot` zero violations).

### - [ ] 1.7 Correlations

- **Now:** Pearson correlation on the raw `data` only.
- **Should be:** The correlation measure is chosen based on feature type and distribution (slide 24), and the result is interpreted.
- **Change:**
  - Compute Pearson **and** Spearman and explain which one fits (skewed size features point toward Spearman).
  - Compute on cleaned data too and compare with raw.
  - Plot a correlation heatmap and write down the multicollinearity finding (important for Task 2).

### - [ ] 1.8 Code style (low priority, helps the defense)

- **Now:** Row-by-row `for` loops, vectors grown with `c(x, ...)`, a loop for the required-columns check.
- **Should be:** Readable code that anyone in the group can explain line by line.
- **Change:** Vectorize, e.g. `setdiff(required_columns, names(data))` and `data$Extent > 1 | data$Extent <= 0`. The CLI argument parsing is optional and can stay or go.

## 2. Missing entirely (required by the task)

### - [ ] 2.1 Missing-value handling

- **Now:** NAs are counted ("left untouched") and then ignored.
- **Should be:** For each column: how many are missing, where, whether they look random, and a justified handling method (slide 17).
- **Change:**
  - Table of NA count and % per column, plus NA count by class.
  - Priority: (1) **recompute from the formula identities** (1.3) when possible, since that gives the exact value; (2) otherwise per-class median; (3) drop the row only if it can't be recovered.
  - Log how each value was filled in.

### - [ ] 2.2 Descriptive statistics

- **Now:** None.
- **Should be:** Statistics chosen by feature type and distribution, overall and **by class** (slide 21: `summary()`, `describeBy()`).
- **Change:** Min, Q1, median, mean, Q3, max, SD, skewness per feature, overall and per class. Every table needs a written conclusion.

### - [ ] 2.3 Visualizations

- **Now:** None.
- **Should be:** Plots that each answer a specific question, numbered, captioned and interpreted (slides 19, 22, 28).
- **Change:** At minimum:
  - Class distribution bar chart (balance)
  - Histograms of key features (skewness)
  - Boxplots **by class** (outliers vs real class differences)
  - Correlation heatmap
  - 1–2 scatter plots of strongly related features coloured by class
  - Save all plots to files so the report can reference them.

### - [ ] 2.4 Scaling

- **Now:** None.
- **Should be:** Scales differ enormously (Area ~10⁴–10⁵ vs ShapeFactor2 ~10⁻³), so compare z-score, Min-Max and Robust scaling and justify the choice (slides 17, 23).
- **Change:** Implement all three, compare resulting statistics and distributions, and pick one with a reason (e.g. Min-Max gets squashed by the extremes we decided to keep).

### - [ ] 2.5 Mandatory comparative experiment

- **Now:** None.
- **Should be:** One preprocessing decision with alternatives compared side by side, explaining *why* the results differ and which suits A21 better (slide 25).
- **Change:** Pick ONE and do it properly:
  - Mean vs median vs formula-based imputation, **or**
  - Z-score vs Min-Max vs Robust scaling, **or**
  - Global IQR vs per-class IQR outlier flagging (we already have the global version).
  - Show before/after statistics, distributions and correlations for each alternative.

### - [ ] 2.6 Before/after comparison

- **Now:** Only row counts before and after.
- **Should be:** A demonstration of how preprocessing changed the data's properties (slide 12).
- **Change:** Statistics table and a few plots of raw vs cleaned data. Also check whether class proportions changed.

### - [ ] 2.7 Reproducibility

- **Now:** The input file is read and not modified (good). Outputs go to `results/`, but the folder isn't created.
- **Should be:** Anyone can run the script on the untouched original file and get identical outputs (slide 28).
- **Change:** `dir.create("results", showWarnings = FALSE)`, `set.seed()` if anything random is added, and one script that produces every table and figure in the report.
