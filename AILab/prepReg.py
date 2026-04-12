import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("indian_roads_dataset.csv")

df = df.drop(['casualties','city','state','date','time','day_of_week','accident_id','longitude','latitude','festival','accident_severity'],axis=1)


numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns


df[numeric_cols]=MinMaxScaler().fit_transform(df[numeric_cols])
df=pd.get_dummies(df,columns=categorical_cols,drop_first=True)
df.to_csv("reg_indian_roads_dataset.csv",index=False)