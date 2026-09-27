import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Settings: adjust to your files and column names ---
RAW_FILE   = "A21.csv"         # original, untouched data
CLEAN_FILE = "A21_clean_TEST.csv"  # TEMP: from ONE_USE_fill_missing_for_testing.py; real file: "../results/A21_without_outliers.csv"  # output of your processing code (cleaned, NOT scaled)
KEY = ["Area", "Perimeter", "AspectRation", "roundness", "Solidity", "ShapeFactor4"]
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

raw = pd.read_csv(RAW_FILE)
df = pd.read_csv(CLEAN_FILE)
num = df.select_dtypes("number").columns
classes = sorted(df["class"].unique())


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close()


# 1. Class balance
counts = df["class"].value_counts()
counts.plot.bar()
plt.ylabel("Count")
save("01_class_counts")

# 2. Missing values in the raw data (skipped if there are none)
na = raw.isna().sum()
if na.sum() > 0:
    na[na > 0].plot.bar()
    plt.ylabel("Missing values")
    save("02_missing_per_feature")

# 3. Distributions of key features
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(axes.flat, KEY):
    ax.hist(df[col], bins=40)
    ax.set_title(col)
save("03_histograms")

# 4. Key features by class (outliers vs. real bean types)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, KEY):
    ax.boxplot([df.loc[df["class"] == c, col] for c in classes])
    ax.set_xticks(range(1, len(classes) + 1), classes, rotation=45)
    ax.set_title(col)
save("04_boxplots_by_class")

# 5. Spearman correlation heatmap
corr = df[num].corr(method="spearman")
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(num)), list(num), rotation=90)
plt.yticks(range(len(num)), list(num))
save("05_spearman_heatmap")

# 6. Major vs. minor axis, coloured by class
plt.figure(figsize=(8, 6))
for c in classes:
    d = df[df["class"] == c]
    plt.scatter(d["MajorAxisLength"], d["MinorAxisLength"], s=5, label=c)
plt.xlabel("MajorAxisLength")
plt.ylabel("MinorAxisLength")
plt.legend()
save("06_scatter_axes")

# 7. Scaling comparison experiment
X = df[num]
scaled = {
    "Z-score": (X - X.mean()) / X.std(),
    "Min-Max": (X - X.min()) / (X.max() - X.min()),
    "Robust":  (X - X.median()) / (X.quantile(0.75) - X.quantile(0.25)),
}
for col in ["Area", "roundness"]:
    plt.figure(figsize=(8, 5))
    plt.boxplot([s[col] for s in scaled.values()])
    plt.xticks([1, 2, 3], list(scaled.keys()))
    plt.title(col)
    save(f"07_scaling_{col}")

# Summary table for the report (Table 4)
summary = pd.concat({k: v[["Area", "roundness"]].describe() for k, v in scaled.items()}, axis=1)
summary.round(3).to_csv(f"{OUT}/07_scaling_summary.csv")

# 8. PCA explained variance: unscaled vs. scaled (why scaling matters)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, data) in zip(axes, [("Unscaled", X), ("Robust-scaled", scaled["Robust"])]):
    ratio = PCA().fit(data).explained_variance_ratio_
    ax.bar(range(1, len(ratio) + 1), ratio)
    ax.set_title(name)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
save("08_pca_variance")

print("Saved plots to", OUT)
