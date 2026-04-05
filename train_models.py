import pandas as pd
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ===============================
# CONFUSION MATRIX HELPER
# ===============================

def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Predicted Legitimate", "Predicted Phishing"],
                yticklabels=["Actual Legitimate", "Actual Phishing"],
                annot_kws={"size": 22, "fontweight": "bold"},
                linewidths=2, linecolor="white",
                ax=ax)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.tick_params(axis="both", labelsize=11)

    legend_text = (
        "Top-Left: True Legitimate — Predicted Legitimate AND it is NOT Phishing (Correct)\n"
        "Top-Right: False Phishing — Predicted Phishing BUT it is NOT Phishing (Error)\n"
        "Bottom-Left: False Legitimate — Predicted Legitimate BUT it is NOT Legitimate (Error)\n"
        "Bottom-Right: True Phishing — Predicted Phishing AND it is NOT Legitimate (Correct)"
    )

    fig.text(0.5, -0.02, legend_text, ha="center", fontsize=9,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {filename}")

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

cm_lr = confusion_matrix(y_test, lr_preds)
print("Confusion Matrix:\n", cm_lr)
plot_confusion_matrix(cm_lr, "Confusion Matrix — Logistic Regression", "confusion_matrix_logistic_regression.png")

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

cm_rf = confusion_matrix(y_test, rf_preds)
print("Confusion Matrix:\n", cm_rf)
plot_confusion_matrix(cm_rf, "Confusion Matrix — Random Forest", "confusion_matrix_random_forest.png")

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



'''
Top-Left (True Negative): Legitimate sites correctly identified.

Bottom-Right (True Positive): Phishing sites correctly identified.

Top-Right (False Positive): Legitimate sites incorrectly flagged as phishing.

Bottom-Left (False Negative): Phishing sites that "slipped through" and were called legitimate.
'''