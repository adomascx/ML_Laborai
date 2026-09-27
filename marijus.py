import pandas as pd
import numpy as np

input_file = "A21.csv"

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

