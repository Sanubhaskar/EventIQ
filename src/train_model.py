import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load cleaned data
df = pd.read_csv("data/cleaned_events.csv")

# Features
features = [
    "event_type",
    "event_cause",
    "priority",
    "requires_road_closure",
    "corridor",
    "zone",
    "hour",
    "day_of_week"
]

df = df[features + ["impact_class"]]

# Fill missing values
df["corridor"] = df["corridor"].fillna("Unknown")
df["zone"] = df["zone"].fillna("Unknown")
df["priority"] = df["priority"].fillna("Low")

# One-hot encoding
X = pd.get_dummies(
    df[features],
    drop_first=True
)

# Target
le = LabelEncoder()

y = le.fit_transform(
    df["impact_class"]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

print(classification_report(
    y_test,
    preds,
    target_names=le.classes_
))
import matplotlib.pyplot as plt

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop Features:")
print(importance.head(20))