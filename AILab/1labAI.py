import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Global_Cybersecurity_Threats_2015-2024.csv")
print("Информация о датасете:")
print(df.info())
print("\n")

print("Типы данных:")
print(df.dtypes)
print("\n")

print("Названия столбцов:")
print(df.columns.tolist())
print("\n")

columns_to_drop = ['Year', 'Country', 'Resolution Time (in Hours)']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print("Количество пропущенных значений по столбцам:")
print(df.isnull().sum())
print("\n")

#Числовые столбцы (сред)
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)

#Категориальные столбцы (мода)
categorical_cols = df.select_dtypes(exclude='number').columns

for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

print("Пропуски после заполнения:")
print(df.isnull().sum())
print("\n")

#Нормализация (Min-Max)
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Данные после нормализации:")
print(df.head())
print("\n")

#One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Данные после OHE-кодирования:")
print(df.head())
print("\n")

#Сохранение
df.to_csv("processed_cybersecurity_dataset.csv", index=False)