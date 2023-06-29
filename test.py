import pickle
import pandas as pd
#from feaures import *
imputer_prob2 = pickle.load(open("features/imputer_prob2.pkl", 'rb'))
one_hot_prob2 = pickle.load(open("features/one_hot_prob2.pkl", 'rb'))
with open("features/columns_one_hot.txt", "r") as f:
    columns_one_hot_list=f.read()
print("imputer_prob2: ",imputer_prob2)
raw_df=pd.read_csv("data/data_phase1_prob2_model-3-4-5-6-7-8.csv")
raw_df=raw_df.drop(columns=["label_model","batch_id","is_drift","prob"])

raw_df=imputer_prob2.transform(raw_df)
raw_df=one_hot_prob2.transform(raw_df)
#get data from raw data in one hot columns
print(raw_df.head())
#exit()
#drop columns not in one hot columns
print(columns_one_hot_list)
#drop columns not in one hot columns in raw data
columns=raw_df.columns.tolist()
#columns not in one hot columns
#drop columns not in one hot columns
columns_not_in_one_hot=[i for i in columns if i not in columns_one_hot_list]
raw_df=raw_df.drop(columns=columns_not_in_one_hot)
