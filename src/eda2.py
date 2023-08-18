import pandas as pd 
import os


data_path="data/predict_incorrect_1.csv"
#find label=0 and predict=1
df=pd.read_csv(data_path)
df_new=df[(df['label']==0) & (df['predict']==1)]
df_new2=df[(df['label']==1) & (df['predict']==0)]
for feature in df.columns[4:]:
    print(feature)
    #print(df_new[feature].value_counts())
    #print(df_new2[feature].value_counts())
    #mean
    print(df_new[feature].mean())
    #max,min
    print(df_new[feature].max())
    print(df_new[feature].min())
    print("************")
    print(df_new2[feature].mean())
    print(df_new2[feature].max())
    print(df_new2[feature].min())
    print("************")