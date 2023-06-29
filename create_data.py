import os 
import numpy as np
import pandas as pd 
data1_path="data/captured_data/phase-1/prob-1/processed"
data2_path="data/captured_data/phase-1/prob-2/processed"
X1=pd.read_parquet(os.path.join(data1_path,"captured_x.parquet"),engine="fastparquet")
X2=pd.read_parquet(os.path.join(data2_path,"captured_x.parquet"),engine="fastparquet")
y1=pd.read_parquet(os.path.join(data1_path,"uncertain_y.parquet"),engine="fastparquet")
y2=pd.read_parquet(os.path.join(data2_path,"uncertain_y.parquet"),engine="fastparquet")

df1=pd.concat([X1,y1],axis=1)
df2=pd.concat([X2,y2],axis=1)
#drop duplicates
df1=df1.drop_duplicates()
df2=df2.drop_duplicates()
print(df1.shape)
print(df2.shape)
df1.to_csv("data/label_data_phase1_prob1.csv",index=False)
df2.to_csv("data/label_data_phase1_prob2.csv",index=False)
    