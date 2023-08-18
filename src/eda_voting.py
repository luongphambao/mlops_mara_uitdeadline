import os 
import pandas as pd 


df1=pd.read_csv('data/data_phase2_prob1_model-1.csv')
df_voting=pd.read_csv('data/data_phase2_prob1_model-voting.csv')
count=0
#find df1!=df_voting
df_diff=df1[df1['label_model']!=df_voting['label_model']]

print(df_diff)
df_diff.to_csv('data/data_phase2_prob1_model-diff.csv',index=False)