from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Settings: adjust to your files and column names ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"


RAW_FILE   = PROJECT_ROOT / "A21.csv"
CLEAN_FILE = RESULTS_DIR / "A21_cleaned.csv"
KEY = ["Area", "Perimeter", "AspectRation", "roundness", "Solidity", "ShapeFactor4"]
OUT = SCRIPT_DIR / "results" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Same colour for each class in every chart
CLASS_COLORS = {"BARBUNYA": "#1f77b4", "BOMBAY": "#d62728", "CALI": "#2ca02c"}

raw = pd.read_csv(RAW_FILE, dtype=str)
df = pd.read_csv(CLEAN_FILE)
num = df.select_dtypes("number").columns
classes = sorted(df["class"].unique())
colors = [CLASS_COLORS[c] for c in classes]

# Raw values as numbers: unit "px" removed, any other text (e.g. "error", "?") becomes NaN
raw_num = raw[list(num)].apply(
    lambda s: pd.to_numeric(s.str.replace(" px", "", regex=False), errors="coerce")
)


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close()


def class_boxplot(ax, data, col):
    box = ax.boxplot([data.loc[data["class"] == c, col] for c in classes], patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_xticks(range(1, len(classes) + 1), classes, rotation=45)
    ax.set_title(col)


# 1. Class balance
counts = df["class"].value_counts().reindex(classes)
counts.plot.bar(color=colors, rot=0)
plt.xlabel("Klasė")
plt.ylabel("Objektų skaičius")
save("01_class_counts")

# 2. Missing or non-numeric values in the raw data, by class
na = raw_num.isna().groupby(raw["class"]).sum().T
na = na[na.sum(axis=1) > 0]
if not na.empty:
    na[classes].plot.bar(stacked=True, color=colors, rot=0)
    plt.xlabel("Požymis")
    plt.ylabel("Trūkstamų reikšmių skaičius")
    plt.legend(title="Klasė")
    save("02_missing_per_feature")

# 3. Distributions of key features
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(axes.flat, KEY):
    ax.hist(df[col], bins=40, color="grey")
    ax.set_title(col)
    ax.set_ylabel("Dažnis")
save("03_histograms")

# 4. Key features by class (outliers vs. real bean types)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, KEY):
    class_boxplot(ax, df, col)
save("04_boxplots_by_class")

# 5. Spearman correlation heatmap
corr = df[num].corr(method="spearman")
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Spearmano koreliacijos koeficientas")
plt.xticks(range(len(num)), list(num), rotation=90)
plt.yticks(range(len(num)), list(num))
save("05_spearman_heatmap")

# 6. Major vs. minor axis, coloured by class
plt.figure(figsize=(8, 6))
for c in classes:
    d = df[df["class"] == c]
    plt.scatter(d["MajorAxisLength"], d["MinorAxisLength"], s=5, label=c, color=CLASS_COLORS[c])
plt.xlabel("MajorAxisLength, px")
plt.ylabel("MinorAxisLength, px")
plt.legend(title="Klasė")
save("06_scatter_axes")

# 7. Scaling comparison experiment
X = df[num]
scaled = {
    "Standartizavimas": (X - X.mean()) / X.std(),
    "Min-Max": (X - X.min()) / (X.max() - X.min()),
    "Robust Scaling":  (X - X.median()) / (X.quantile(0.75) - X.quantile(0.25)),
}
for col in ["Area", "roundness"]:
    plt.figure(figsize=(8, 5))
    plt.boxplot([s[col] for s in scaled.values()])
    plt.xticks([1, 2, 3], list(scaled.keys()))
    plt.ylabel("Reikšmė po mastelio keitimo")
    plt.title(col)
    save(f"07_scaling_{col}")

# Summary table for the report
summary = pd.concat({k: v[["Area", "roundness"]].describe() for k, v in scaled.items()}, axis=1)
summary.round(3).to_csv(f"{OUT}/07_scaling_summary.csv")

# 8. PCA explained variance: unscaled vs. scaled (why scaling matters)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, data) in zip(axes, [("Be mastelio keitimo", X), ("Robust Scaling", scaled["Robust Scaling"])]):
    ratio = PCA().fit(data).explained_variance_ratio_
    ax.bar(range(1, len(ratio) + 1), ratio)
    ax.set_title(name)
    ax.set_xlabel("Pagrindinė komponentė")
    ax.set_ylabel("Paaiškintos dispersijos dalis")
save("08_pca_variance")

# 9. Before / after cleaning for the features changed by logical-bound corrections
CHANGED = ["ConvexArea", "Solidity", "Compactness"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, CHANGED):
    ax.boxplot([raw_num[col].dropna(), df[col]])
    ax.set_xticks([1, 2], ["Prieš", "Po"])
    ax.set_title(col)
save("09_before_after")

# Descriptive statistics table for the report
desc = X.describe().T
desc["skew"] = X.skew()
desc["IQR"] = desc["75%"] - desc["25%"]
desc.to_csv(f"{OUT}/10_descriptive_stats.csv", float_format="%.4g")
df.groupby("class")[list(num)].median().T.to_csv(f"{OUT}/10_median_by_class.csv", float_format="%.4g")

print("Saved plots to", OUT)
