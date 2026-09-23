from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "frailty_raw.csv"
PROCESSED_PATH = DATA_DIR / "frailty_processed.csv"

# Read the raw CSV data
df = pd.read_csv(RAW_PATH)

# Unit standardization
# 1. Convert height from inches to meters
df["Height_m"] = df["Height_in"] * 0.0254

# 2. Convert weight from pounds to kilograms
df["Weight_kg"] = df["Weight_lb"] * 0.45359237

# Feature engineering
# 1. Calculate BMI
df["BMI"] = (df["Weight_kg"] / (df["Height_m"] ** 2)).round(2)

# 2. Create age groups
df["AgeGroup"] = pd.cut(df["Age_yr"],
    [-float("inf"), 29, 45, 60, float("inf")],
    labels=["<30", "30–45", "46–60", ">60"])

# Categorical - numerical encoding
# 1. Convert frailty from Y/N to 1/0
df["Frailty_binary"] = (
    df["Frailty"]
    .map({"Y": 1, "N": 0})
    .astype("int8"))
# 2. Create one-hot age-group columns
dummies = pd.get_dummies(
    df["AgeGroup"],
    prefix="AgeGroup",
    dtype="int8")
# 3. Add the new age-group columns to the dataset
df = pd.concat([df, dummies], axis=1)

# EDA & Reporting
# 1. Save the processed dataset to a new CSV file
df.to_csv(PROCESSED_PATH, index=False)

# 2. Summarize the dataset
summary = df.select_dtypes("number").agg(["mean", "median", "std"]).T
print(summary)

# 3. Calculate the grip-strength/frailty correlation
corr = df["Grip_kg"].corr(df["Frailty_binary"])
print("Grip/Frailty correlation:", corr)