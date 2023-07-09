import os 
import json 
import pandas as pd
import numpy as np
import sys 

phase_id="phase-2"
prob_id="prob-1"
data_path="data/captured_data/"+phase_id+"/"+prob_id+"/"
save_path="data/"+"data_captured_"+phase_id+"_"+prob_id+".csv"
list_df=[]
for file in os.listdir(data_path):
    if file.endswith(".json"):
        print(file)
        file_path=data_path+file
        with open(file_path,"r") as f:
            data=json.load(f)
        rows=data["rows"]
        columns=data["columns"]
        df=pd.DataFrame(rows,columns=columns)
        
        list_df.append(df)
df=pd.concat(list_df)
#reindex columns feature1 to feature41
df=df.reindex(columns=["feature"+str(i) for i in range(1,42)])
#drop duplicates
df=df.drop_duplicates()
df.to_csv(save_path,index=False)