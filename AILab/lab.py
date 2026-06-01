import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("Class_Social_media_impact_on_life.csv")
target = "Overall_Impact_Positive"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Random Forest
RF = RandomForestClassifier(oob_score=True)
RF.fit(X_train, y_train)
y_pred_test_rf = RF.predict(X_test)
y_proba_rf = RF.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
roc_auc = auc(fpr, tpr)

print("\nRandom Forest")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_test_rf))
print("\nОтчет:\n", classification_report(y_test, y_pred_test_rf))
print("\nROC AUC:", roc_auc_score(y_test, y_proba_rf))
print("OOB score:", RF.oob_score_)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая (Random Forest)')
plt.legend(loc="lower right")
plt.savefig("ROCcurveRF.png")
plt.show()

# AdaBoost
AB = AdaBoostClassifier(random_state=42)
AB.fit(X_train, y_train)

y_pred_test_ab = AB.predict(X_test)
y_proba_ab = AB.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba_ab)
roc_auc = auc(fpr, tpr)

print("\nAdaBoost")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_test_ab))
print("\nОтчет:\n", classification_report(y_test, y_pred_test_ab))
print("ROC AUC:", roc_auc_score(y_test, y_proba_ab))

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая (AdaBoost)')
plt.legend(loc="lower right")
plt.savefig("ROCcurveAB.png")
plt.show()

# Gradient Boosting
GBC = GradientBoostingClassifier(random_state=42)
GBC.fit(X_train, y_train)

y_pred_test_gbc = GBC.predict(X_test)
y_proba_gbc = GBC.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba_gbc)
roc_auc = auc(fpr, tpr)

print("\nGradientBoosting")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_test_gbc))
print("\nОтчет:\n", classification_report(y_test, y_pred_test_gbc))
print("ROC AUC:", roc_auc_score(y_test, y_proba_gbc))

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая (GradientBoosting)')
plt.legend(loc="lower right")
plt.savefig("ROCcurveGBC.png")
plt.show()