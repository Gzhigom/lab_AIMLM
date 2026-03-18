import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Метрики
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report
)

df = pd.read_csv('processed_cybersecurity_dataset.csv')

print("Размер датасета:", df.shape)
print(df.head())

##################################
# РЕГРЕССИЯ
##################################

print("\n РЕГРЕССИЯ")

# Целевая переменная
y_reg = df['Financial Loss (in Million $)']

# Признаки
X_reg = df.drop(columns=['Financial Loss (in Million $)'])

# Разделение
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=19
)

# Модель
reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

# Предсказание
y_pred_r = reg_model.predict(X_test_r)

# Оценка
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_r, y_pred_r)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

##################################
# КЛАССИФИКАЦИЯ
##################################

print("\n КЛАССИФИКАЦИЯ")

# Целевая переменная (бинарная)
y_clf = df['Defense Mechanism Used_VPN'].astype(int)

# Признаки
X_clf = df.drop(columns=['Defense Mechanism Used_VPN'])

# Разделение
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=19
)

# Модель
clf_model = LogisticRegression(max_iter=1000,class_weight='balanced')
clf_model.fit(X_train_c, y_train_c)

# Предсказание
y_pred_c = clf_model.predict(X_test_c)

# Оценка
accuracy = accuracy_score(y_test_c, y_pred_c)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c))