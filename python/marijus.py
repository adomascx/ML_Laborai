from pathlib import Path
import pandas as pd
import numpy as np
import re

pd.set_option("display.precision", 17) #kad neapvalintu po kablelio

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


#duomenu sutvarkymas (pavertimas i NaN arba sutvarkymas normaliai..)
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

numeric_data = pd.DataFrame(
    np.nan,
    index=data.index,
    columns=numeric_columns
)

change_log = pd.DataFrame(
    "",
    index=data.index,
    columns=numeric_columns
)

for column in numeric_columns:
    parsed = data[column].map(parse_numeric)

    numeric_data[column] = parsed.map(lambda x: x[0])
    change_log[column] = parsed.map(lambda x: x[1])

print (numeric_data.head())

#sutvarkytos eilutes, pereinam prie  tikros uzduoties
#logiskai galimu ribu patikrinimas

#area: Area = π / 4 * EquivDiameter²
# π / 4 * MajorAxisLength * MinorAxisLength neveikia
#Perimeter - not calculable
#AspectRation = MajorAxisLength/MinorAxisLength
#Eccentricity = √(1-(Major-Minor)²)
#ConvexArea = Area / Solidity (jei gerai surasyti sitie)
#EquivDiameter = √(4 × Area / π)
#Extent - non calculable
#Solidity = Area / ConvexArea
#Roundness = 4 × π × Area / Perimeter²
#Compactness - non calculable
#ShapeFactor1 = MajorAxisLength / Area
#ShapeFactor2 - non calculable
#ShapeFactor3 - non calculable
#ShapeFactor4 - non calculable