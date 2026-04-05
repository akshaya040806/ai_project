import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42

# ===============================
# LOAD PREPROCESSED DATA
# ===============================

required = ["X_train_scaled.pkl", "X_test_scaled.pkl", "y_train.pkl", "y_test.pkl"]
for f in required:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found. Run preprocess.py first.")

X_train = joblib.load("X_train_scaled.pkl")
X_test  = joblib.load("X_test_scaled.pkl")
y_train = joblib.load("y_train.pkl")
y_test  = joblib.load("y_test.pkl")

print(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}")

# Check if dataset was flagged as imbalanced
IMBALANCED = joblib.load("imbalanced_flag.pkl") if os.path.exists("imbalanced_flag.pkl") else False
class_weight = "balanced" if IMBALANCED else None
if IMBALANCED:
    print("Imbalanced dataset detected — using class_weight='balanced'")

# ===============================
# HELPER: print metrics
# ===============================

def print_metrics(name, y_true, y_pred, y_prob=None):
    print(f"\n{name} Results:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  AUC-ROC           {auc:.4f}")

# ===============================
# MODEL 1 — Logistic Regression
# ===============================

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=3000, class_weight=class_weight)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]
print_metrics("Logistic Regression", y_test, lr_preds, lr_probs)
joblib.dump(lr, "logistic_model.pkl")

# ===============================
# MODEL 2 — Random Forest
# ===============================

print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight=class_weight
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]
print_metrics("Random Forest", y_test, rf_preds, rf_probs)
joblib.dump(rf, "random_forest_model.pkl")

# ===============================
# MODEL 3 — XGBoost (Optional)
# ===============================

if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost...")
    xgb_weight = (y_train == 0).sum() / (y_train == 1).sum() if IMBALANCED else 1
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=xgb_weight
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    print_metrics("XGBoost", y_test, xgb_preds, xgb_probs)
    joblib.dump(xgb, "xgboost_model.pkl")
else:
    print("\nXGBoost not installed — skipping.")

print("\n✅ Training Complete. Models saved.")
