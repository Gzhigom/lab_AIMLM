
import pandas as pd


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

#############################################################################################
#                                         ПРОПУСКИ
#############################################################################################


# Поиск

print("Количество пропущенных значений по столбцам:")
print(df.isnull().sum())
print("\n")


# Заполнение


# Числовые столбцы (сред)
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)

# Категориальные столбцы (мода)
categorical_cols = df.select_dtypes(exclude='number').columns

for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

print("Пропуски после заполнения:")
print(df.isnull().sum())
print("\n")

#############################################################################################
#                             Нормализация (Min-Max)
#############################################################################################

for col in numeric_cols:
    min_value = df[col].min()
    max_value = df[col].max()

    #  деление на 0
    if max_value != min_value:
        df[col] = (df[col] - min_value) / (max_value - min_value)
    else:
        df[col] = 0

print("Данные после нормализации:")
print(df.head())
print("\n")

#############################################################################################
#                                     One-Hot Encoding
#############################################################################################

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Данные после OHE-кодирования:")
print(df.head())
print("\n")

#############################################################################################
# 7. Сохранение обработанного датасета
#############################################################################################

df.to_csv("processed_cybersecurity_dataset.csv", index=False)

print("Обработка завершена. Файл сохранён как processed_cybersecurity_dataset.csv")