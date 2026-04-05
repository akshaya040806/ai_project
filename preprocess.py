import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===============================
# CONFIG
# ===============================

DATASET = "dataset_phishing.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

INFERENCE_FEATURES = [
    "length_url",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_eq",
    "nb_slash",
    "nb_www",
    "nb_com",
    "https_token",
    "ip",
    "nb_subdomains",
]



# LOAD
print("=" * 50)
print("STEP 1: Loading dataset")
print("=" * 50)

if not os.path.exists(DATASET):
    raise FileNotFoundError(f"{DATASET} not found.")

df = pd.read_csv(DATASET)
print(f"  Rows loaded       : {len(df)}")
print(f"  Columns loaded    : {len(df.columns)}")

# VALIDATE TARGET
print("\nSTEP 2: Validating target column")

if "status" not in df.columns:
    raise ValueError("Dataset must contain a 'status' column.")

print(f"  Raw status values : {df['status'].unique().tolist()}")

#encoding
if df["status"].dtype == object:
    df["status"] = df["status"].str.lower().str.strip()

    valid_labels = {"phishing": 1, "legitimate": 0}
    df["status"] = df["status"].map(valid_labels)

    if df["status"].isna().sum() > 0:
        raise ValueError("❌ Unknown labels found in 'status' column")

    print("  Encoded string labels → 0 (legitimate), 1 (phishing)")
else:
    df["status"] = df["status"].str.lower().str.strip()
    df["status"] = df["status"].map({"legitimate": 0,"phishing": 1})

    if df["status"].isna().sum() > 0:
         raise ValueError("Unknown values in status column")
         print("  Status already numeric")

    print(f"  Encoded values     : {df['status'].unique().tolist()}")
    y = df["status"]


# CLASS BALANCE
print("\nSTEP 3: Checking class balance")

counts = y.value_counts()
total = len(y)

phishing_count = counts.get(1, 0)
legit_count = counts.get(0, 0)

ratio = max(phishing_count, legit_count) / max(min(phishing_count, legit_count), 1)

print(f"  Legitimate (0)    : {legit_count} ({legit_count/total*100:.1f}%)")
print(f"  Phishing   (1)    : {phishing_count} ({phishing_count/total*100:.1f}%)")
print(f"  Imbalance ratio   : {ratio:.2f}:1")

IMBALANCED = ratio > 3.0

if IMBALANCED:
    print("  ⚠ Dataset is imbalanced — use class_weight='balanced'")

joblib.dump(IMBALANCED, "imbalanced_flag.pkl")



# DROPPING UNUSED COLUMNS
print("\nSTEP 4: Dropping non-feature columns")

cols_to_drop = ["status"]
if "url" in df.columns:
    cols_to_drop.append("url")

df = df.drop(columns=cols_to_drop)

print(f"  Dropped           : {cols_to_drop}")
print(f"  Remaining columns : {len(df.columns)}")



# HANDLING MISSING VALUES
print("\nSTEP 5: Handling missing values")

null_counts = df.isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]

if len(cols_with_nulls) > 0:
    print(f"  Columns with nulls: {cols_with_nulls.to_dict()}")

    for col in cols_with_nulls.index:
        if df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed '{col}' with median={median_val:.2f}")
        else:
            df = df.drop(columns=[col])
            print(f"  Dropped non-numeric column: '{col}'")
else:
    print("  No missing values")



# FEATURE ALIGNMENT
print("\nSTEP 6: Aligning features")

missing = [f for f in INFERENCE_FEATURES if f not in df.columns]
extra = [c for c in df.columns if c not in INFERENCE_FEATURES]

if missing:
    print(f"  Missing features → filling with 0: {missing}")
    for col in missing:
        df[col] = 0

print(f"  Dropping {len(extra)} extra columns")

X = df[INFERENCE_FEATURES].copy()

print(f"  Final features ({len(X.columns)}): {X.columns.tolist()}")



# CLIP OUTLIERS
print("\nSTEP 7: Clipping outliers")

clip_rules = {
    "length_url": 500,
    "nb_subdomains": 10,
    "nb_dots": 20,
    "nb_slash": 30,
}

for col, upper in clip_rules.items():
    if col in X.columns:
        X[col] = X[col].clip(upper=upper)



# HANDING NUMERICS
print("\nSTEP 8: Ensuring numeric features")

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)




# TRAIN/TEST SPLIT
print("\nSTEP 9: Train/Test split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  X_train: {X_train.shape}")
print(f"  X_test : {X_test.shape}")



# SCALING METRICS
print("\nSTEP 10: Scaling")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nSTEP 11: Saving files")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(INFERENCE_FEATURES, "features.pkl")
joblib.dump(X_train_scaled, "X_train_scaled.pkl")
joblib.dump(X_test_scaled, "X_test_scaled.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")

print("[+] All files saved")
print("PREPROCESSING COMPLETE")




