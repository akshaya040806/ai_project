import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ===============================
# CONFIG
# ===============================

DATASET = "dataset_phishing.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ===============================
# LOAD DATA
# ===============================

print("Loading dataset...")

if not os.path.exists(DATASET):
    raise FileNotFoundError(f"{DATASET} not found")

df = pd.read_csv(DATASET)

print("Total rows:", len(df))

# ===============================
# VALIDATION
# ===============================

if "status" not in df.columns:
    raise ValueError("Dataset must contain a 'status' column")

df.dropna(inplace=True)

# Target
y = df["status"]

# Remove URL column if present
if "url" in df.columns:
    df = df.drop(columns=["url"])

# Features
X = df.drop(columns=["status"])

print("Total features:", len(X.columns))

# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Data split complete.")

# ===============================
# SCALE FEATURES
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# ===============================
# MODEL 1 — Logistic Regression
# ===============================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(max_iter=3000)
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)

print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_preds))

joblib.dump(lr, "logistic_model.pkl")

# ===============================
# MODEL 2 — Random Forest
# ===============================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE
)

rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_preds))

joblib.dump(rf, "random_forest_model.pkl")

# ===============================
# MODEL 3 — XGBoost (Optional)
# ===============================

if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost...")

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    xgb.fit(X_train, y_train)

    xgb_preds = xgb.predict(X_test)

    print("\nXGBoost Results:")
    print(classification_report(y_test, xgb_preds))

    joblib.dump(xgb, "xgboost_model.pkl")
else:
    print("\nXGBoost not installed — skipping.")

print("\n✅ Training Complete. Models and scaler saved.")
