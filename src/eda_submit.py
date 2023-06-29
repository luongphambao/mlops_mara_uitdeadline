import pandas as pd 
import numpy as np 

csv_path="data/data_phase1_prob1.csv"
df=pd.read_csv(csv_path)
#distribution of batch_id not sorted
print(df["batch_id"].value_counts())
print("check unique batch_id")
print(len(df["batch_id"].unique()))