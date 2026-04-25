import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("Class_Social_media_impact_on_life.csv")
target = "Overall_Impact_Positive"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

AB = AdaBoostClassifier(learning_rate=0.6, n_estimators=1000, random_state=42)
AB.fit(X_train, y_train)

y_pred_test_ab = AB.predict(X_test)
y_proba_ab = AB.predict_proba(X_test)[:, 1]

print("\nAdaBoost")
print("Точность:", accuracy_score(y_test, y_pred_test_ab))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_test_ab))
print("\nОтчет:\n", classification_report(y_test, y_pred_test_ab))
print("ROC AUC:", roc_auc_score(y_test, y_proba_ab))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba_ab)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая (AdaBoost)')
plt.legend(loc="lower right")
plt.savefig("ROCcurveAB.png")
plt.show()

GBC = GradientBoostingClassifier(random_state=42)
GBC.fit(X_train, y_train)

y_pred_test_gbc = GBC.predict(X_test)
y_proba_gbc = GBC.predict_proba(X_test)[:, 1]

print("\n GradientBoosting ")
print("Точность:", accuracy_score(y_test, y_pred_test_gbc))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_test_gbc))
print("\nОтчет:\n", classification_report(y_test, y_pred_test_gbc))
print("ROC AUC:", roc_auc_score(y_test, y_proba_gbc))

fpr, tpr, _ = roc_curve(y_test, y_proba_gbc)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая (GradientBoosting)')
plt.legend(loc="lower right")
plt.savefig("ROCcurveGBC.png")
plt.show()