import os 
import numpy as np 
import pandas as pd 
import json
import sys


data_path1="data/data_phase1_prob1_light_gbm.csv"
data_path2="data/data_phase1_prob1_xgb.csv"

df1=pd.read_csv(data_path1)
df2=pd.read_csv(data_path2)

#find label_model different in 2 dataframe
df1=df1.drop(columns=["batch_id","is_drift"])
# change label_model with prob<0.9 to 0 if label_model=1 and 1 if label_model=0
df1.loc[(df1["label_model"]==1) & (df1["prob"]<0.9),"label_model"]=0
df1.loc[(df1["label_model"]==0) & (df1["prob"]<0.9),"label_model"]=1
df1.to_csv("data/data_phase1_prob1_light_gbm_change.csv",index=False)
print(len(df1))
print(len(df2))
# #find label_model not equal
# df1_not_equal=df1[df1["label_model"]!=df2["label_model"]]
# df2_not_equal=df2[df1["label_model"]!=df2["label_model"]]
# #create new df1 with label df1,prob df1, label df2, prob df2
# df1_not_equal=df1_not_equal.reset_index(drop=True)
# df2_not_equal=df2_not_equal.reset_index(drop=True)
# df1_not_equal["label_model_xgb"]=df2_not_equal["label_model"]
# df1_not_equal["prob_xgb"]=df2_not_equal["prob"]
# df1_not_equal["label_model_light_gbm"]=df1_not_equal["label_model"]
# df1_not_equal["prob_light_gbm"]=df1_not_equal["prob"]
# df1_not_equal=df1_not_equal.drop(columns=["label_model","prob"])
# df1_not_equal.to_csv("data/compare_prob.csv",index=False)
# print(len(df1_not_equal))
# for index,row in df1_not_equal.iterrows():
#     print(row["label_model_xgb"],row["label_model_light_gbm"])
#     print(row["prob_xgb"],row["prob_light_gbm"])
#     #print(index)
#     print("different")
#     #exit()