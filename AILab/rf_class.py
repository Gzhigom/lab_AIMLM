import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, auc, roc_curve, roc_auc_score
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("Class_Social_media_impact_on_life.csv")
target = "Overall_Impact_Positive"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

RF = RandomForestClassifier(oob_score=True)
RF.fit(X_train, y_train)
y_pred_test_rf = RF.predict(X_test)

y_pred_test = RF.predict(X_test)
y_proba = RF.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

accuracy = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test)
roc_score = roc_auc_score(y_test, y_proba)
print("\naccuracy:", accuracy)
print("\nConfusion matrix:\n", cm)
print("\nОтчет:\n", report)
print("\nROC AUC:", roc_score)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая ')
plt.legend(loc="lower right")
plt.savefig("ROCcurveRF.png")
plt.show()
print("OOB score:", RF.oob_score_)
