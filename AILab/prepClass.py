import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("Social_media_impact_on_life.csv")
df = df.drop(['Student_ID','Mental_Health_Score','Country'],axis=1)


numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns


df[numeric_cols]=MinMaxScaler().fit_transform(df[numeric_cols])
df=pd.get_dummies(df,columns=categorical_cols,drop_first=True)
df = df.drop(['Overall_Impact_Neutral'],axis=1)
df.to_csv("Class_Social_media_impact_on_life.csv",index=False)