from pathlib import Path
import pandas as pd
import numpy as np
import re

pd.set_option("display.precision", 17) #kad neapvalintu skaičių po kablelio display

script_dir = Path(__file__).resolve().parent

input_file = script_dir / "A21.csv"
output_file = script_dir / "A21_cleaned.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Input file '{input_file}' not found.")

data = pd.read_csv(
    input_file,
    dtype=str
)

print(data.head())

original_data = data.copy()

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
data.iloc[:, 1] = (
    data.iloc[:, 1]
    .str.replace(r"\s*px$", "", regex=True)
    .str.strip()
)
def parse_numeric(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan
    
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"(?:\s+[A-Za-z%µ°]+)?",
        value
    )

    if match:
        return float(match.group(1))

    return np.nan

numeric_data = data[numeric_columns].map(parse_numeric)

print("?" * 70)
print("?" * 70)

print(numeric_data.head(10))

numeric_data.to_csv(output_file, index=False)