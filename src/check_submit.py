import pandas as  pd 
import numpy as np 
import json 


data_path1="data/data_phase1_prob1.csv"
data_path2="data/data_phase1_prob2.csv"

label_path1="data/label_data_phase1_prob1.csv"
label_path2="data/label_data_phase1_prob2.csv"

df1=pd.read_csv(data_path1)
df2=pd.read_csv(data_path2)
label1=pd.read_csv(label_path1)
label2=pd.read_csv(label_path2)
dict_map_feature5_label1={}
dict_map_feature5_label2={}
#round 10 decimal places
label1["feature5"]=label1["feature5"].apply(lambda x:round(x,10))
for index,row in label1.iterrows():
    dict_map_feature5_label1[(row["feature5"])]=int(row["label"])
for index,row in label2.iterrows():
    dict_map_feature5_label2[(row["feature5"])]=int(row["label"])
#print(dict_map_feature5_label1["24.60962048866008"])
print(len(dict_map_feature5_label2))
unique_batch_id1=df1["batch_id"].unique()
print(len(unique_batch_id1))
unique_batch_id2=df2["batch_id"].unique()
#get label for batch_id 
dict_result1={}
dict_result2={}
count=0
feature5_save_path="data/feature5_prob1.json"
feature5_save_path2="data/feature5_prob2.json"
with open(feature5_save_path,"w") as f:
    json.dump(dict_map_feature5_label1,f)
with open(feature5_save_path2,"w") as f:
    json.dump(dict_map_feature5_label2,f)
# for batch_id in unique_batch_id1:
#     #get label for batch_id by using dict_map_feature5_label1
#     #get all data with batch_id
#     data=df1[df1["batch_id"]==batch_id]
#     feature5=data["feature5"].values.tolist()
#     label=[dict_map_feature5_label1[str(round(i,10))] for i in feature5]
#     #print(label)
#     print(len(label))
#     #exit()
#     dict_result1[batch_id]=label
#     count+=1
#     print(count)

# for batch_id in unique_batch_id2:
#     #get label for batch_id by using dict_map_feature5_label2
#     data=df2[df2["batch_id"]==batch_id]
#     feature5=data["feature5"].values.tolist()
#     label=[dict_map_feature5_label2[str(round(i,10))] for i in feature5]
#     #print(label)
#     #exit()
#     dict_result2[batch_id]=label
# #convert int64 to str
# print(len(dict_result1))
# print(len(dict_result2))
# dict_result1={str(k):(v) for k,v in dict_result1.items()}
# dict_result2={str(k):(v) for k,v in dict_result2.items()}
# #save dict_result1 and dict_result2
# with open("data/dict_result1.json","w") as f:
#     json.dump(dict_result1,f)
# with open("data/dict_result2.json","w") as f:
#     json.dump(dict_result2,f)