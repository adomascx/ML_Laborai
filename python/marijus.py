from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path(__file__).resolve().parent

input_file = script_dir / "A21.csv"
output_file = script_dir / "A21_cleaned.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Input file '{input_file}' not found.")

# 2ia baigiau keisti

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

numeric_data = data[numeric_columns].apply( 
    pd.to_numeric,
    errors="coerce"
)

