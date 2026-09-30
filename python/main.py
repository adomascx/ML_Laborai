from pathlib import Path

import pandas as pd
import numpy as np
import re



# Pandas settings for it to not round values shown in the display
pd.set_option("display.precision", 17)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
results_dir = script_dir / "results"

input_file = project_root / "A21.csv"
output_file = results_dir / "A21_cleaned.csv"
original_data_file = results_dir / "A21_original_copy.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Input file '{input_file}' not found.")

data = pd.read_csv(
    input_file,
    dtype=str
)

data.to_csv(original_data_file, index=False)

original_data = data.copy()


# Duplicate countand removal

duplicate_mask = data.duplicated(keep="first")                      # Binary mask of duplicates for rows
duplicate_rows_count = duplicate_mask.sum()                         # Calculates the number of TRUE values
duplicates_removed_count = data.duplicated(keep="first").sum()      # Calculates the amount of duplicates that will be removed

duplicate_rows = data[duplicate_mask]
duplicate_rows.to_csv(
    results_dir / "duplicate_rows.csv",
    index=False
)

data = data.drop_duplicates(ignore_index=True)                      # Removes duplicates and resets the index

# Save duplicate statistics
duplicate_report = pd.DataFrame({
    "duplicate_rows_found": [duplicate_rows_count],
    "duplicate_rows_removed": [duplicates_removed_count],
    "rows_before_removal": [len(original_data)],
    "rows_after_removal": [len(data)]
})

duplicate_report.to_csv(
    results_dir / "duplicate_report.csv",
    index=False
)



# List of numberic columns to be cleaned
numeric_columns = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]



# function to parse numeric values from strings, handling various formats and units
def parse_numeric(value):
    if pd.isna(value):
        return np.nan, ""

    original = str(value)
    value = original.strip()

    if value == "":
        return np.nan, ""

    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"(?:\s+[A-Za-z%µ°]+)?",
        value
    )

    if not match:
        return np.nan, ""

    number = float(match.group(1))

    if value != match.group(1):
        change = f"{original!r} -> {number}"
    elif original != value:
        change = f"{original!r} -> {value!r}"
    else:
        change = ""

    return number, change

numeric_data = data[numeric_columns].map(parse_numeric)



# Extract only the numeric value from (number, change)
numeric_data = numeric_data.map(lambda result: result[0])



# First check for missing values in the original data
numeric_missing_raw = (
    numeric_data.isna()
    .sum()
    .rename_axis("column")
    .reset_index(name="missing_count")
)
numeric_missing_raw.to_csv(
    results_dir / "numeric_missing_values_raw.csv",
    index=False
)



# Eccentricity checks and calculations, validates minor and major axes for them
# Eccentricity = sqrt(1 - (minor_axis / major_axis)^2)
major_axis = numeric_data["MajorAxisLength"]
minor_axis = numeric_data["MinorAxisLength"]

valid_values = (
    major_axis.notna()
    & minor_axis.notna()
    & (major_axis > 0)
    & (minor_axis >= 0)
    & (minor_axis <= major_axis)
)

calculated_eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)

missing_eccentricity = (
    numeric_data["Eccentricity"].isna()
    & valid_values
)

numeric_data.loc[
    missing_eccentricity,
    "Eccentricity"
] = calculated_eccentricity[missing_eccentricity]



# Roundness checks and calculations, validates area and perimeter for them
# Roundness = (4 * pi * area) / (perimeter^2)
area = numeric_data["Area"]
perimeter = numeric_data["Perimeter"]

valid_roundness_values = (
    area.notna()
    & perimeter.notna()
    & (area > 0)
    & (perimeter > 0)
)

calculated_roundness = (4 * np.pi * area) / (perimeter ** 2)

missing_roundness = (
    numeric_data["roundness"].isna()
    & valid_roundness_values
)

numeric_data.loc[
    missing_roundness,
    "roundness"
] = calculated_roundness[missing_roundness]



# Extent checks and calculations, it is non calculable hence it will be calculated by class averages
extent_class_averages = numeric_data["Extent"].groupby(data["class"]).transform("mean")
numeric_data["Extent"] = numeric_data["Extent"].fillna(extent_class_averages)



# ShapeFactor4 checks and calculations, since it is non calculable it will be calculated by class averages
shapefactor4_class_averages = numeric_data["ShapeFactor4"].groupby(data["class"]).transform("mean")
numeric_data["ShapeFactor4"] = numeric_data["ShapeFactor4"].fillna(shapefactor4_class_averages)



# Finished filling missing values exported to CSV

data[numeric_columns] = numeric_data
data.to_csv(output_file, index=False)

# Second check for any missing values to validate that the cleaning process worked
numeric_missing_clean = (
    numeric_data.isna()
    .sum()
    .rename_axis("column")
    .reset_index(name="missing_count")
)

numeric_missing_clean.to_csv(
    results_dir / "numeric_missing_values_clean.csv",
    index=False
)