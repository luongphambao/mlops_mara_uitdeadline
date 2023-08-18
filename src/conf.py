import pandas as pd 
import numpy as np


data_path="correct_data.csv"
data_path1="data/data_phase3_prob1_model-1.csv"
data=pd.read_csv(data_path)
#drop duplicate
data.drop_duplicates(inplace=True)
data1=pd.read_csv(data_path1)
data1.drop_duplicates(inplace=True)


#find prob <0.8
data1_more_prob=data1[data1['prob']>0.9]
data_less_prob=data[data['prob']<0.9]
print((len(data_less_prob)))
data1_less_prob=data1[data1['prob']<0.9]
print((len(data1_less_prob)))
#convert df to list rows
data_less_prob_list=data_less_prob.values.tolist()
data1_less_prob_list=data1_less_prob.values.tolist()
#find data1 not in data
data1_not_in_data=[]
for i in data1_less_prob_list:
    if i not in data_less_prob_list:
        data1_not_in_data.append(i)
#convert list to df
data1_not_in_data=pd.DataFrame(data1_not_in_data,columns=data1.columns)
#change label_model=1-label_model
data1_not_in_data['label_model']=1-data1_not_in_data['label_model']

#merge data1_not_in_data and data_less_prob and data1_more_prob
df=pd.concat([data1_not_in_data,data_less_prob,data1_more_prob],ignore_index=True)
df.to_csv("final_data.csv",index=False)
