from pathlib import Path

import pandas as pd
import numpy as np
import re



# Pandas settings for it to not round values shown in the display
pd.set_option("display.precision", 17)

script_dir = Path(__file__).resolve().parent

input_file = script_dir / "A21.csv"
output_file = script_dir / "A21_cleaned.csv"
original_data_file = script_dir / "A21_original_copy.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Input file '{input_file}' not found.")

data = pd.read_csv(
    input_file,
    dtype=str
)

data.to_csv(original_data_file, index=False)

original_data = data.copy()



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
    script_dir / "numeric_missing_values_raw.csv",
    index=False
)



# Eccentricity checks and calculations, validates minor and major axes for them
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


numeric_data.to_csv(output_file, index=False) 



# Second check for any missing values to validate that the cleaning process worked
numeric_missing_clean = (
    numeric_data.isna()
    .sum()
    .rename_axis("column")
    .reset_index(name="missing_count")
)

numeric_missing_clean.to_csv(
    script_dir / "numeric_missing_values_clean.csv",
    index=False
)