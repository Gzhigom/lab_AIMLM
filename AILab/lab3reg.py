import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

df = pd.read_csv("reg_indian_roads_dataset.csv")
X = df.drop("risk_score",axis=1)
y=df["risk_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

dt_regressor_model = DecisionTreeRegressor(max_depth=5)
dt_regressor_model.fit(X_train, y_train)
y_pred_test=dt_regressor_model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("Mean Square Error",MSE)
print("Root MSE",RMSE)
print("Mean Absolute Error",MAE)
print("r^2",r2)

plt.figure(figsize=(30, 15))
tree.plot_tree(dt_regressor_model,
               feature_names=X.columns.tolist(),
               filled=True,
               fontsize=4,
               max_depth=4)
plt.title("дерево решений")
plt.savefig("regr_tree.png")
plt.show()