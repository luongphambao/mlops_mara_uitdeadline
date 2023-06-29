import pandas as pd
import json 
data_labeling_prob1=pd.read_csv("data/label_data_phase1_prob1.csv")
data_labeling_prob2=pd.read_csv("data/label_data_phase1_prob2.csv")

dict_result1={}
dict_result2={}
for index,row in data_labeling_prob1.iterrows():
    row=row.to_list()
    feature=row[:-1]
    label=row[-1]
    dict_result1[str(feature)]=label
    #print(feature)
    #exit()
print(len(dict_result1))
json_path="data/curl/phase-1/prob-1/ed681ea6-a548-4dba-8e5d-1cde677ea05f.json"
with open(json_path) as f:
    test_data=json.load(f)
test_data=test_data["rows"]
test_data=pd.DataFrame(test_data)
#drop duplicates
test_data=test_data.drop_duplicates()
print(test_data.shape)
#drop last 2 columns
print(test_data.shape)
#test_data=test_data.drop(columns=[16,17])
count=0 
print(len(test_data))
for index,row in test_data.iterrows():
    row=row.to_list()
    row=str(row)
    #print(row)
    if row in dict_result1:
        #print(dict_result1[row])
        count+=1
print(count)