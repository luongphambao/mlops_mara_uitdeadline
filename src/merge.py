import os 
import pandas as  pd 

df0=pd.read_csv("data/predict_incorrect_0.csv")
df1=pd.read_csv("data/predict_incorrect_1.csv")
df2=pd.read_csv("data/predict_incorrect_2.csv")
df3=pd.read_csv("data/predict_incorrect_3.csv")
df4=pd.read_csv("data/predict_incorrect_4.csv")
#merge all the dataframes
df=pd.concat([df0,df1,df2,df3,df4],ignore_index=True)
df.to_csv("data/predict_incorrect.csv",index=False)