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

# These are the ONLY features that extract_features() in ai.py can compute.
# Training must use exactly these columns — nothing more — to avoid
# zero-padding mismatches at inference time.
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

# ===============================
# STEP 1 — LOAD
# ===============================

print("=" * 50)
print("STEP 1: Loading dataset")
print("=" * 50)

if not os.path.exists(DATASET):
    raise FileNotFoundError(f"{DATASET} not found. Place it in the same directory.")

df = pd.read_csv(DATASET)
print(f"  Rows loaded       : {len(df)}")
print(f"  Columns loaded    : {len(df.columns)}")

# ===============================
# STEP 2 — VALIDATE TARGET
# ===============================

print("\nSTEP 2: Validating target column")

if "status" not in df.columns:
    raise ValueError("Dataset must contain a 'status' column.")

# Show raw values before encoding
print(f"  Raw status values : {df['status'].unique().tolist()}")

# Encode: 1 = phishing, 0 = legitimate
# Handles both string labels and already-numeric labels
if df["status"].dtype == object:
    df["status"] = (df["status"].str.lower().str.strip() == "phishing").astype(int)
    print("  Encoded string labels to 0/1")
else:
    df["status"] = df["status"].astype(int)
    print("  Status column already numeric — kept as-is")

y = df["status"]

# ===============================
# STEP 3 — CHECK CLASS BALANCE
# ===============================

print("\nSTEP 3: Checking class balance")

counts = y.value_counts()
total = len(y)
phishing_count = counts.get(1, 0)
legit_count = counts.get(0, 0)
ratio = max(phishing_count, legit_count) / max(min(phishing_count, legit_count), 1)

print(f"  Legitimate (0)    : {legit_count} ({legit_count/total*100:.1f}%)")
print(f"  Phishing   (1)    : {phishing_count} ({phishing_count/total*100:.1f}%)")
print(f"  Imbalance ratio   : {ratio:.2f}:1")

if ratio > 3.0:
    print("  WARNING: Dataset is imbalanced (ratio > 3:1).")
    print("           Add class_weight='balanced' to your models in train_models.py")
    IMBALANCED = True
else:
    print("  Class balance is acceptable.")
    IMBALANCED = False

# Save flag so train_models.py can read it
joblib.dump(IMBALANCED, "imbalanced_flag.pkl")

# ===============================
# STEP 4 — DROP NON-FEATURE COLUMNS
# ===============================

print("\nSTEP 4: Dropping non-feature columns")

cols_to_drop = ["status"]
if "url" in df.columns:
    cols_to_drop.append("url")

df = df.drop(columns=cols_to_drop)
print(f"  Dropped           : {cols_to_drop}")
print(f"  Remaining columns : {len(df.columns)}")

# ===============================
# STEP 5 — HANDLE MISSING VALUES
# ===============================

print("\nSTEP 5: Handling missing values")

null_counts = df.isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]

if len(cols_with_nulls) > 0:
    print(f"  Columns with nulls: {cols_with_nulls.to_dict()}")
    # Impute numeric columns with column median (safer than drop for URL features)
    for col in cols_with_nulls.index:
        if df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed '{col}' with median={median_val:.2f}")
        else:
            df.drop(columns=[col], inplace=True)
            print(f"  Dropped non-numeric column with nulls: '{col}'")
else:
    print("  No missing values found.")

# ===============================
# STEP 6 — ALIGN TO INFERENCE FEATURES
# ===============================

print("\nSTEP 6: Aligning features to inference feature set")

missing_from_dataset = [f for f in INFERENCE_FEATURES if f not in df.columns]
extra_in_dataset = [c for c in df.columns if c not in INFERENCE_FEATURES]

if missing_from_dataset:
    print(f"  WARNING: These inference features are missing from dataset: {missing_from_dataset}")
    print("           They will be filled with 0 — consider expanding extract_features() in ai.py")
    for col in missing_from_dataset:
        df[col] = 0

print(f"  Dropping {len(extra_in_dataset)} extra training-only columns (not computable at inference)")
X = df[INFERENCE_FEATURES].copy()

print(f"  Final feature count: {len(X.columns)}")
print(f"  Features           : {X.columns.tolist()}")

# ===============================
# STEP 7 — CLIP OUTLIERS
# ===============================

print("\nSTEP 7: Clipping outliers")

clip_rules = {
    "length_url"    : 500,
    "nb_subdomains" : 10,
    "nb_dots"       : 20,
    "nb_slash"      : 30,
}

for col, upper in clip_rules.items():
    if col in X.columns:
        before_max = X[col].max()
        X[col] = X[col].clip(upper=upper)
        after_max = X[col].max()
        if before_max != after_max:
            print(f"  Clipped '{col}': max {before_max} -> {after_max}")
        else:
            print(f"  '{col}': no clipping needed (max={before_max})")

# ===============================
# STEP 8 — VERIFY NUMERIC DTYPES
# ===============================

print("\nSTEP 8: Verifying all features are numeric")

non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"  WARNING: Non-numeric columns found: {non_numeric}")
    print("           Attempting conversion with pd.to_numeric...")
    for col in non_numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    print("  Conversion complete.")
else:
    print("  All features are numeric — OK")

# ===============================
# STEP 9 — TRAIN / TEST SPLIT
# ===============================

print("\nSTEP 9: Stratified train/test split (80/20)")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")
print(f"  y_train dist  : {y_train.value_counts().to_dict()}")
print(f"  y_test  dist  : {y_test.value_counts().to_dict()}")

# ===============================
# STEP 10 — SCALE FEATURES
# ===============================

print("\nSTEP 10: Fitting StandardScaler on training data only")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # transform only on test (no leakage)

print("  Scaler fitted on X_train — X_test transformed without re-fitting")
print(f"  Mean (first 3 features)  : {scaler.mean_[:3].round(4)}")
print(f"  Std  (first 3 features)  : {scaler.scale_[:3].round(4)}")

# ===============================
# STEP 11 — SAVE ARTIFACTS
# ===============================

print("\nSTEP 11: Saving preprocessing artifacts")

joblib.dump(scaler,           "scaler.pkl")
joblib.dump(INFERENCE_FEATURES, "features.pkl")
joblib.dump(X_train_scaled,   "X_train_scaled.pkl")
joblib.dump(X_test_scaled,    "X_test_scaled.pkl")
joblib.dump(y_train,          "y_train.pkl")
joblib.dump(y_test,           "y_test.pkl")

print("  Saved: scaler.pkl")
print("  Saved: features.pkl  (use this in ai.py instead of scaler.feature_names_in_)")
print("  Saved: X_train_scaled.pkl")
print("  Saved: X_test_scaled.pkl")
print("  Saved: y_train.pkl")
print("  Saved: y_test.pkl")

# ===============================
# SUMMARY
# ===============================

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)
print(f"  Final training samples : {X_train_scaled.shape[0]}")
print(f"  Final test samples     : {X_test_scaled.shape[0]}")
print(f"  Features used          : {len(INFERENCE_FEATURES)}")
print(f"  Class imbalanced       : {IMBALANCED}")
if IMBALANCED:
    print("  ACTION NEEDED: Add class_weight='balanced' to models in train_models.py")
print("\nRun train_models.py next — it will load the saved .pkl files.")
