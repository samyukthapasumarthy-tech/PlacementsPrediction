import pandas as pd
df=pd.read_csv(r"C:\Users\samyu\Desktop\ipl dataset(ml)\matches.csv")
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())


