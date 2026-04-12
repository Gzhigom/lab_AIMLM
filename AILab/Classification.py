import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


df = pd.read_csv("Class_Social_media_impact_on_life.csv").dropna()
target = "Overall_Impact_Positive"
X = df.drop(columns=[target])
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=19, stratify=y
)

model = LogisticRegression(
    solver="saga",
    l1_ratio=0.5,
    max_iter=1000,
    class_weight="balanced",
    C=1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.7).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))