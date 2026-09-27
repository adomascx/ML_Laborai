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


#sutvarkytos eilutes, pereinam prie  tikros uzduoties
#logiskai galimu ribu patikrinimas

#area: Area = π / 4 * EquivDiameter²
# π / 4 * MajorAxisLength * MinorAxisLength neveikia
#Perimeter - not calculable
#AspectRation = MajorAxisLength/MinorAxisLength
#Eccentricity = √(1-(MinorAxisLength/MajorAxisLength)²)
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

#Patikrinamos loginės ribos prieš taisymą.

print("\n--- Logical-bound check BEFORE corrections ---")

# Visi skaitiniai dydžiai turi būti > 0

for column in numeric_columns:
    violations = (
        numeric_data[column].notna()
        & (numeric_data[column] <= 0)
    ).sum()

    if violations > 0:
        print(f"{column}: {violations} violations")
    else:
        print(f"{column}: OK")

# Tam tikri dydžiai turi būti <= 1

bounded_columns = [
    "Eccentricity",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness"
]

print("\nBounded columns <= 1:")

for column in bounded_columns:
    violations = (
        numeric_data[column].notna()
        & (numeric_data[column] > 1)
    ).sum()

    if violations > 0:
        print(f"{column}: {violations} violations")
    else:
        print(f"{column}: OK")

# AspectRation turi būti >= 1

violations = (
    numeric_data["AspectRation"].notna()
    & (numeric_data["AspectRation"] < 1)
).sum()

if violations > 0:
    print(f"\nAspectRation >= 1: {violations} violations")
else:
    print("\nAspectRation >= 1: OK")

# ConvexArea turi būti >= Area

violations = (
    numeric_data["ConvexArea"].notna()
    & numeric_data["Area"].notna()
    & (numeric_data["ConvexArea"] < numeric_data["Area"])
).sum()

if violations > 0:
    print(f"ConvexArea >= Area: {violations} violations")
else:
    print("ConvexArea >= Area: OK")

# MajorAxisLength turi būti >= MinorAxisLength

violations = (
    numeric_data["MajorAxisLength"].notna()
    & numeric_data["MinorAxisLength"].notna()
    & (
        numeric_data["MajorAxisLength"]
        < numeric_data["MinorAxisLength"]
    )
).sum()

if violations > 0:
    print(
        f"MajorAxisLength >= MinorAxisLength: "
        f"{violations} violations"
    )
else:
    print("MajorAxisLength >= MinorAxisLength: OK")


#sukuriamas laikinas logas, kad neskaiciuotu iki siol padarytu changes.
logical_bound_log = pd.DataFrame(
    "",
    index=numeric_data.index,
    columns=numeric_columns
)

print("\n--- Applying logical-bound corrections ---")

# ConvexArea >= Area

mask = (
    numeric_data["ConvexArea"].notna()
    & numeric_data["Area"].notna()
    & (numeric_data["ConvexArea"] < numeric_data["Area"])
    & numeric_data["Solidity"].notna()
    & (numeric_data["Solidity"] > 0)
)

for index in numeric_data.index[mask]:
    old_value = numeric_data.at[index, "ConvexArea"]

    new_value = (
        numeric_data.at[index, "Area"]
        / numeric_data.at[index, "Solidity"]
    )

    numeric_data.at[index, "ConvexArea"] = new_value

    message = (
        f"logical-bound correction: {old_value} -> {new_value}; "
        f"calculated as Area / Solidity"
    )

    logical_bound_log.at[index, "ConvexArea"] = message

    if change_log.at[index, "ConvexArea"].strip():
        change_log.at[index, "ConvexArea"] += "; " + message
    else:
        change_log.at[index, "ConvexArea"] = message

# Solidity <= 1

mask = (
    numeric_data["Solidity"].notna()
    & (numeric_data["Solidity"] > 1)
    & numeric_data["Area"].notna()
    & numeric_data["ConvexArea"].notna()
    & (numeric_data["ConvexArea"] > 0)
)

for index in numeric_data.index[mask]:
    old_value = numeric_data.at[index, "Solidity"]

    new_value = (
        numeric_data.at[index, "Area"]
        / numeric_data.at[index, "ConvexArea"]
    )

    numeric_data.at[index, "Solidity"] = new_value

    message = (
        f"logical-bound correction: {old_value} -> {new_value}; "
        f"calculated as Area / ConvexArea"
    )

    logical_bound_log.at[index, "Solidity"] = message

    if change_log.at[index, "Solidity"].strip():
        change_log.at[index, "Solidity"] += "; " + message
    else:
        change_log.at[index, "Solidity"] = message

# Compactness <= 1

mask = (
    numeric_data["Compactness"].notna()
    & (numeric_data["Compactness"] > 1)
)

for index in numeric_data.index[mask]:
    class_name = data.at[index, "class"]

    class_mean = numeric_data.loc[
        data["class"] == class_name,
        "Compactness"
    ].mean()

    if pd.notna(class_mean):
        old_value = numeric_data.at[index, "Compactness"]
        new_value = class_mean

        numeric_data.at[index, "Compactness"] = new_value

        message = (
            f"logical-bound correction: {old_value} -> {new_value}; "
            f"replaced with {class_name} class mean"
        )

        logical_bound_log.at[index, "Compactness"] = message

        if change_log.at[index, "Compactness"].strip():
            change_log.at[index, "Compactness"] += "; " + message
        else:
            change_log.at[index, "Compactness"] = message

# AspectRation >= 1

mask = (
    numeric_data["AspectRation"].notna()
    & (numeric_data["AspectRation"] < 1)
    & numeric_data["MajorAxisLength"].notna()
    & numeric_data["MinorAxisLength"].notna()
    & (numeric_data["MinorAxisLength"] > 0)
)

for index in numeric_data.index[mask]:
    old_value = numeric_data.at[index, "AspectRation"]

    new_value = (
        numeric_data.at[index, "MajorAxisLength"]
        / numeric_data.at[index, "MinorAxisLength"]
    )

    numeric_data.at[index, "AspectRation"] = new_value

    message = (
        f"logical-bound correction: {old_value} -> {new_value}; "
        f"calculated as MajorAxisLength / MinorAxisLength"
    )

    logical_bound_log.at[index, "AspectRation"] = message

    if change_log.at[index, "AspectRation"].strip():
        change_log.at[index, "AspectRation"] += "; " + message
    else:
        change_log.at[index, "AspectRation"] = message

#Pranešame apie pakeitimus dėl loginių ribų.

logical_changes = (
    logical_bound_log[numeric_columns]
    .apply(lambda column: column.str.strip().ne(""))
)

print("\n--- Logical-bound corrections by column ---")
print(
    logical_changes.sum()[
        logical_changes.sum() > 0
    ]
)

print(
    f"\nTotal logical-bound corrections: "
    f"{logical_changes.sum().sum()}"
)


#Patikrinamos loginės ribos po taisymo.

print("\n--- Logical-bound check AFTER corrections ---")

# Visi skaitiniai dydžiai turi būti > 0

for column in numeric_columns:
    violations = (
        numeric_data[column].notna()
        & (numeric_data[column] <= 0)
    ).sum()

    if violations > 0:
        print(f"{column}: {violations} remaining violations")
    else:
        print(f"{column}: OK")

# Tam tikri dydžiai turi būti <= 1

print("\nBounded columns <= 1:")

for column in bounded_columns:
    violations = (
        numeric_data[column].notna()
        & (numeric_data[column] > 1)
    ).sum()

    if violations > 0:
        print(f"{column}: {violations} remaining violations")
    else:
        print(f"{column}: OK")

# AspectRation turi būti >= 1

violations = (
    numeric_data["AspectRation"].notna()
    & (numeric_data["AspectRation"] < 1)
).sum()

if violations > 0:
    print(
        f"\nAspectRation >= 1: "
        f"{violations} remaining violations"
    )
else:
    print("\nAspectRation >= 1: OK")

# ConvexArea turi būti >= Area

violations = (
    numeric_data["ConvexArea"].notna()
    & numeric_data["Area"].notna()
    & (numeric_data["ConvexArea"] < numeric_data["Area"])
).sum()

if violations > 0:
    print(
        f"ConvexArea >= Area: "
        f"{violations} remaining violations"
    )
else:
    print("ConvexArea >= Area: OK")

# MajorAxisLength turi būti >= MinorAxisLength

violations = (
    numeric_data["MajorAxisLength"].notna()
    & numeric_data["MinorAxisLength"].notna()
    & (
        numeric_data["MajorAxisLength"]
        < numeric_data["MinorAxisLength"]
    )
).sum()

if violations > 0:
    print(
        f"MajorAxisLength >= MinorAxisLength: "
        f"{violations} remaining violations"
    )
else:
    print("MajorAxisLength >= MinorAxisLength: OK")


# Laikinas logas nebereikalingas.

del logical_bound_log


# išskirtys:
#Patikrinamos išorinės išskirtys pagal Q1 - 3*IQR ir Q3 + 3*IQR.

print("\n--- Outlier check (Q1 - 3*IQR / Q3 + 3*IQR) ---")

outlier_thresholds = {}

for column in numeric_columns:
    q1 = numeric_data[column].quantile(0.25)
    q3 = numeric_data[column].quantile(0.75)
    iqr = q3 - q1

    lower_threshold = q1 - 3 * iqr
    upper_threshold = q3 + 3 * iqr

    outlier_thresholds[column] = (
        lower_threshold,
        upper_threshold
    )

    lower_outliers = (
        numeric_data[column].notna()
        & (numeric_data[column] < lower_threshold)
    ).sum()

    upper_outliers = (
        numeric_data[column].notna()
        & (numeric_data[column] > upper_threshold)
    ).sum()

    print(
        f"{column}: "
        f"Q1 = {q1}, "
        f"Q3 = {q3}, "
        f"IQR = {iqr}, "
        f"lower threshold = {lower_threshold}, "
        f"upper threshold = {upper_threshold}, "
        f"lower outliers = {lower_outliers}, "
        f"upper outliers = {upper_outliers}"
    )


#Patikrinama, kokiose klasėse yra viršutiniai outlieriai.

print("\n--- Upper outliers by class ---")

for column in numeric_columns:
    lower_threshold, upper_threshold = outlier_thresholds[column]

    mask = (
        numeric_data[column].notna()
        & (numeric_data[column] > upper_threshold)
    )

    if mask.any():
        print(f"\n{column}:")
        print(
            data.loc[mask, "class"]
            .value_counts()
            .to_string()
        )


#Patikrinama, kiek eilučių turi bent vieną viršutinį outlierį.

upper_outlier_rows = pd.Series(
    False,
    index=numeric_data.index
)

for column in numeric_columns:
    lower_threshold, upper_threshold = outlier_thresholds[column]

    mask = (
        numeric_data[column].notna()
        & (numeric_data[column] > upper_threshold)
    )

    upper_outlier_rows |= mask

print(
    f"Unique rows with at least one upper outlier: "
    f"{upper_outlier_rows.sum()}"
)


#Patikrinama, kiek viršutinių outlierių turi kiekviena eilutė.

upper_outlier_count = pd.Series(
    0,
    index=numeric_data.index
)

for column in numeric_columns:
    lower_threshold, upper_threshold = outlier_thresholds[column]

    mask = (
        numeric_data[column].notna()
        & (numeric_data[column] > upper_threshold)
    )

    upper_outlier_count += mask.astype(int)

print("\n--- Upper outliers per row ---")
print(
    upper_outlier_count[
        upper_outlier_count > 0
    ].value_counts().sort_index()
)


#Sukuriamas laikinas outlierių žurnalas.

outlier_log = pd.Series(
    "",
    index=numeric_data.index,
    dtype="object"
)

for column in numeric_columns:
    lower_threshold, upper_threshold = outlier_thresholds[column]

    mask = (
        numeric_data[column].notna()
        & (numeric_data[column] > upper_threshold)
    )

    for index in numeric_data.index[mask]:
        if outlier_log.at[index]:
            outlier_log.at[index] += "; " + column
        else:
            outlier_log.at[index] = column

print("\n--- Upper outlier flags ---")
print(
    f"Rows with upper outliers: "
    f"{(outlier_log != '').sum()}"
)

#Viršutinės ribos išskirtys įrašomos į bendrą pakeitimų žurnalą.

for index in numeric_data.index[outlier_log != ""]:
    columns = outlier_log.at[index].split("; ")

    for column in columns:
        message = "upper outlier: Q3 + 3*IQR"

        if change_log.at[index, column].strip():
            change_log.at[index, column] += "; " + message
        else:
            change_log.at[index, column] = message


#Laikinas išskirčių žurnalas nebereikalingas.

del outlier_log

#Sukuriama galutinė išvalyta duomenų lentelė.

cleaned_data = data.copy()

for column in numeric_columns:
    cleaned_data[column] = numeric_data[column]

#Sujungiami visi pakeitimai į vieną change_log stulpelį.

cleaned_data["change_log"] = change_log.apply(
    lambda row: "; ".join(
        f"{column}: {row[column]}"
        for column in numeric_columns
        if row[column].strip()
    ),
    axis=1
)

#Tušti change_log įrašai paliekami tušti.

cleaned_data["change_log"] = cleaned_data["change_log"].replace("", np.nan)

print("\n--- Final dataset ---")
print(cleaned_data.head())

print(
    f"\nRows: {len(cleaned_data)}"
)

print(
    f"Columns: {len(cleaned_data.columns)}"
)

#Išsaugomas galutinis išvalytas CSV failas.

output_file = "A21_cleaned.csv"

cleaned_data.to_csv(
    output_file,
    index=False
)

print(f"\nCleaned data saved to: {output_file}")


#Sukuriamas atskiras change log tekstinis failas.

change_log_file = "A21_change_log.txt"

with open(change_log_file, "w", encoding="utf-8") as file:
    for index in cleaned_data.index:
        log_entry = cleaned_data.at[index, "change_log"]

        if pd.notna(log_entry):
            file.write(
                f"Row {index}: {log_entry}\n"
            )

print(f"Change log saved to: {change_log_file}")