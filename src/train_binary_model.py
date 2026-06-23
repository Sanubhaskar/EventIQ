import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# =========================
# Load Data
# =========================

df = pd.read_csv("data/cleaned_events.csv")

# =========================
# Create Binary Target
# =========================

df["risk_level"] = df["duration_minutes"].apply(
    lambda x: "HighRisk" if x >= 180 else "LowRisk"
)

print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())

# =========================
# Features
# =========================

features = [
    "event_type",
    "event_cause",
    "priority",
    "requires_road_closure",
    "corridor",
    "zone",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "event_cause_freq",
    "corridor_freq",
    "is_non_corridor"
]

# Keep only required columns
df = df[features + ["risk_level"]]

# =========================
# Handle Missing Values
# =========================

df["corridor"] = df["corridor"].fillna("Unknown")
df["zone"] = df["zone"].fillna("Unknown")
df["priority"] = df["priority"].fillna("Low")

# =========================
# One-Hot Encoding
# =========================

X = pd.get_dummies(
    df[features],
    drop_first=True
)

# =========================
# Encode Target
# =========================

le = LabelEncoder()

y = le.fit_transform(
    df["risk_level"]
)

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Model
# =========================

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
# =========================
# Predictions
# =========================
probs = model.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score

print(
    "auc score:" ,roc_auc_score(y_test, probs)
)
preds = (probs >= 0.3).astype(int)
##preds = model.predict(X_test)
results = X_test.copy()

results["original_index"] = X_test.index
results["actual"] = y_test
results["predicted"] = preds
results["probability"] = probs
results.to_csv(
    "data/error_analysis.csv",
    index=False
)

print("\nSaved error analysis file.")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

print(scores)
print("Mean AUC:", scores.mean())

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        preds,
        target_names=le.classes_
    )
)

# =========================
# Feature Importance
# =========================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 Important Features:\n")
print(importance.head(20))

# =========================
# Save Model
# =========================

import pickle

with open("models/eventiq_binary_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel saved successfully!")

from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))