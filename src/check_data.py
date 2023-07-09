import pandas as pd 
import numpy as np


data_path="data/data_phase2_prob1_model-1_0.95.csv"
data_path="data/data_phase2_prob1_model-1_1.csv"
data=pd.read_csv(data_path)
# #find index with prob <=0.6 and label=1
# index=data[(data["prob"]<=0.7) & (data["label_model"]==1)].index
# print(len(index))
# #change label to 0
# data.loc[index,"label_model"]=0
data.to_csv("data/data_phase2_prob1_model-1.csv",index=False)
