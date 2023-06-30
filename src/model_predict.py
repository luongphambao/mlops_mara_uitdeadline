import argparse
import logging
import os
import random
import time
import pickle
import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel
#IMPOrt auc score
from sklearn.metrics import roc_auc_score
import sys  
import numpy as np
import json 
try:
    from problem_config import ProblemConst, create_prob_config
    from raw_data_processor import RawDataProcessor
    from utils import AppConfig, AppPath
except:
    from src.problem_config import ProblemConst, create_prob_config
    from src.raw_data_processor import RawDataProcessor
    from src.utils import AppConfig, AppPath
from pydantic import BaseSettings

class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
            print(self.config)
            print("load config")
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        #self.model = mlflow.pyfunc.load_model(model_uri)
        self.model = mlflow.sklearn.load_model(model_uri)
        self.scaler=pickle.load(open("features/robust_scaler_phase-1_prob-1.pkl", 'rb')) if self.prob_config.prob_id=="prob-1" else pickle.load(open("features/robust_scaler_phase-1_prob-2.pkl", 'rb'))
        #self.test_data=pd.read_csv("data/label_data_phase1_prob1.csv") if self.prob_config.prob_id=="prob-1" else pd.read_csv("data/label_data_phase1_prob2.csv")
        #self.submit_data=pd.read_csv("data/data_phase1_prob1.csv") if self.prob_config.prob_id=="prob-1" else pd.read_csv("data/data_phase1_prob2.csv")
    def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        #time.sleep(0.02)
        return random.choice([0, 1])

    def test(self, raw_df):
        start_time = time.time()
        #get data except label
        label=raw_df["label"]
        df=raw_df.drop(columns=["label"])
        #raw_df.to_csv(str(self.prob_config.captured_data_dir)+"/"+str(data.id)+".csv",index=False)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        feature_df=self.scaler.transform(feature_df)
        prediction = self.model.predict(feature_df)
        confidence_score=np.max(prediction,axis=1)

        print("AUC score: ",roc_auc_score(label, prediction))
        #save incorrect prediction
        incorrect_prediction=raw_df[label!=prediction]
        name="data/incorrect_prediction_"+str(self.prob_config.phase_id)+"_"+str(self.prob_config.prob_id)+"_"+str(self.config["model_name"])+"_"+str(self.config["model_version"])+".csv"
        incorrect_prediction.to_csv(name,index=False)
    def predict_df(self,raw_df,save_path=None,prob=False):
        #df_data=raw_df.drop(columns=["batch_id","is_drift"])
        #df_data=raw_df.drop(columns=["batch_id","is_drift","prob"])
        df_data=raw_df
        if "batch_id" in df_data.columns:
            df_data=df_data.drop(columns=["batch_id"])
        if "is_drift" in df_data.columns:
            df_data=df_data.drop(columns=["is_drift"])
        if "label_model" in raw_df.columns:
            df_data=df_data.drop(columns=["label_model"])
        if "prob" in raw_df.columns:
            df_data=df_data.drop(columns=["prob"])
        print(df_data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=df_data,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        feature_df=self.scaler.transform(feature_df)
        prediction = self.model.predict(feature_df)
        prediction=[int(i) for i in prediction]
        if prob==True:
            prob_prediction=self.model.predict_proba(feature_df)
            prob_prediction=np.max(prob_prediction,axis=1).tolist()
            raw_df["prob"]=prob_prediction
        raw_df["label_model"]=prediction
        raw_df.to_csv(save_path,index=False)
        return raw_df
    def predict_submit_batch(self,raw_df):
        
        df_data=raw_df.drop(columns=["batch_id","is_drift","label_model","prob"])
        
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=df_data,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        feature_df=self.scaler.transform(feature_df)
        prediction = self.model.predict(feature_df)
        raw_df["label"]=prediction
        unique_batch_id=raw_df["batch_id"].unique()
        result_dict={}
        for batch_id in unique_batch_id:
            df_batch=raw_df[raw_df["batch_id"]==batch_id]
            result_dict[str(batch_id)]=df_batch["label"].tolist()
            result_dict[str(batch_id)]=[int(i) for i in result_dict[str(batch_id)]]
        save_json_path="data/"+str(self.prob_config.phase_id)+"_"+str(self.prob_config.prob_id)+"_"+str(self.config["model_name"])+"_"+str(self.config["model_version"])+".json"
        with open(save_json_path,"w") as f:
            json.dump(result_dict,f)
        return result_dict
if __name__=="__main__":
    csv_path="data/data_phase1_prob2_model-3.csv"
    df = pd.read_csv(csv_path)
    model_name="model-3.yaml"
    model_config_path="data/model_config/phase-1/prob-2/"+model_name
    predictor=ModelPredictor(model_config_path)
    predictor.predict_df(df,"data/data_phase1_prob1_"+model_name.replace("yaml","csv"),prob=True)
    
    #for i in range(1,5):
        
       # model_name="model-"+str(i)+".yaml"
       # model_config_path="data/model_config/phase-1/prob-1/"+model_name
       # print(model_config_path)
       # model_predictor=ModelPredictor(model_config_path)
       # csv_model_path="data/data_phase1_prob1_"+model_name.replace("yaml","csv")
        #model_predictor.predict_df(df,csv_model_path,prob=True)
    #print("done")
