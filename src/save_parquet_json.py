import pandas as pd 
import os
import json
parquet_folder_path="data/captured_data/phase-1/prob-2/"
save_json_path="data/curl/phase-1/prob-2/"
columns= [
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
    "feature11",
    "feature12",
    "feature13",
    "feature14",
    "feature15",
    "feature16",
    "feature17",
    "feature18",
    "feature19",
    "feature20",
  ]
for file in os.listdir(parquet_folder_path):
    if file.endswith(".parquet"):
        dict_result = {}
        df = pd.read_parquet(parquet_folder_path+file)
        #get df not have last 2 columns
        
        df = df.iloc[:, :-2]
        print(df.shape)
        #save df to 2d array
        data=df.to_numpy()
        data = data.tolist()
        #convert value to float
        data=[[float(j) for j in i] for i in data]
        #convert float to int if value is int
        data=[[int(j) if j.is_integer() else j for j in i] for i in data]
        id = file.split(".")[0]
        dict_result["id"] = id
        dict_result["rows"] = data
        dict_result["columns"] = columns
        
        #save to json
        with open(save_json_path+id+".json", 'w') as outfile:
            json.dump(dict_result, outfile)
        
        
