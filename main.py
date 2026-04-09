# ==============================
# 1) Import Libraries
# ==============================
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# ==============================
# 2) Create Data
# ==============================
data = {
    "usage": [2, 5, 8, 1, 7, 3, 9, 2],
    "complaints": [8, 2, 1, 9, 2, 7, 1, 8],
    "label": [1, 0, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# ==============================
# 3) Prepare Data
# ==============================
X = df[["usage", "complaints"]]
y = df["label"]

# ==============================
# 4) Train Model
# ==============================
model = LogisticRegression()
model.fit(X, y)

# ==============================
# 5) Default Prediction & Evaluation
# ==============================
predictions = model.predict(X)

print("=== Default Model ===")
print("Accuracy:", accuracy_score(y, predictions))
print("Confusion Matrix:\n", confusion_matrix(y, predictions))
print("Precision:", precision_score(y, predictions))
print("Recall:", recall_score(y, predictions))

# ==============================
# 6) Probability + Threshold
# ==============================
probs = model.predict_proba(X)

threshold = 0.5

custom_predictions = (probs[:, 1] > threshold).astype(int)

print(f"\n=== Custom Model (threshold={threshold}) ===")
print("Accuracy:", accuracy_score(y, custom_predictions))
print("Confusion Matrix:\n", confusion_matrix(y, custom_predictions))
print("Precision:", precision_score(y, custom_predictions))
print("Recall:", recall_score(y, custom_predictions))