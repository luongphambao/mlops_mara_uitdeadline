import os 
import evidently
import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.options import DataDriftOptions
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns,TestColumnDrift
import json 
import time
import warnings
warnings.filterwarnings("ignore")
class  DriftPredictor:
    def __init__(self, phase_id="phase-3",prob_id="prob-1"):
        self.phase_id=phase_id
        self.prob_id=prob_id
        self.data_path="data/raw_data/{}/{}".format(phase_id,prob_id)+"/raw_train.parquet"
        data_drift_dataset_tests = TestSuite(tests=[
                                    TestColumnDrift(column_name="feature3"),
                                    ])
        df = pd.read_parquet(self.data_path)
        df=df.drop(["label"],axis=1)
        #select 5000 samples
        df=df.sample(n=10000,random_state=1)
        self.df=df
        self.model_drift=data_drift_dataset_tests
    def predict_data_drift(self,test_df): 
        detect_drift=self.model_drift
        detect_drift.run(reference_data=self.df, current_data=test_df)
        data_drift_data_dict=detect_drift.as_dict()
        descriptions=data_drift_data_dict["tests"][0]["description"]
        #print(descriptions)
        descriptions=float(descriptions.split(" ")[8][:-1])
        is_drift=1 if descriptions>0.1 else 0
        return is_drift
# data_path="data/raw_data/phase-3/prob-1/raw_train.parquet"
# df = pd.read_parquet(data_path)
# df=df.drop(["label"],axis=1)
# print(df.head())
# import json 
# test_path="data/captured_data/phase-3/prob-1/ff5127cf-8291-4f3f-90ca-2a0457fe13e9.json"
# with open(test_path) as f:
#     test_data = json.load(f)
# rows = test_data["rows"]
# columns = test_data["columns"]
# test_df = pd.DataFrame(rows, columns=columns)
# #sort df columns feature 1 to feature 41
# test_df=test_df.reindex(['feature{}'.format(i) for i in range(1,42)], axis=1)
# print(test_df.head())

# #dataset-level tests
# data_drift_dataset_tests = TestSuite(tests=[
#     TestNumberOfDriftedColumns(),
#     TestShareOfDriftedColumns(),    
# ])

# data_drift_dataset_tests.run(reference_data=df, current_data=test_df)
# data_drift_data_dict=data_drift_dataset_tests.as_dict()
# save_path=os.path.basename(test_path).replace(".json","_data_drift.json")
# with open(save_path,"w") as f:
#     json.dump(data_drift_data_dict,f)
# print(data_drift_data_dict["tests"][0]["description"])
if __name__=="__main__":
    drift_predictor=DriftPredictor()
    count=0
    for file in os.listdir("data/captured_data/phase-3/prob-1/"):
        data_path="data/captured_data/phase-3/prob-1/"+file
        if data_path.endswith(".json")==False:
            continue
        with open(data_path) as f:
            test_data = json.load(f)
        try:
            rows = test_data["rows"]
            columns = test_data["columns"]
        except:
            continue
        test_df = pd.DataFrame(rows, columns=columns)
        test_df=test_df.reindex(['feature{}'.format(i) for i in range(1,42)], axis=1)
        start_time=time.time()
        is_drift=drift_predictor.predict_data_drift(test_df)
        print("is_drift: ",is_drift)
        if count==2 and is_drift==1:
            print("file: ",file)
            exit()
        #print("",descriptions)
        print("time elapsed: ",time.time()-start_time)
        count+=is_drift

    print("Percentage of drift: ",count/len(os.listdir("data/captured_data/phase-3/prob-1/")))
        