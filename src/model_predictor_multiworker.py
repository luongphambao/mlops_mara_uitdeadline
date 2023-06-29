import argparse
import logging
import os
import random
import time

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel

import sys  
try:
    from problem_config import ProblemConst, create_prob_config
    from raw_data_processor import RawDataProcessor
    from utils import AppConfig, AppPath
except:
    from src.problem_config import ProblemConst, create_prob_config
    from src.raw_data_processor import RawDataProcessor
    from src.utils import AppConfig, AppPath
from pydantic import BaseSettings






PREDICTOR_API_PORT = 8000
app = FastAPI()
model_prob1="data/model_config/phase-1/prob-1/model-1.yaml"
model_prob2="data/model_config/phase-1/prob-2/model-1.yaml"
feature_df_list=[]
data_id_list=[]
capture_data_path_list=[]
class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
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
        self.model = mlflow.pyfunc.load_model(model_uri)

    def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        time.sleep(0.01)
        return random.choice([0, 1])

    def predict(self, data: Data):
        start_time = time.time()

        # preprocess

        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        #print(raw_df)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        logging.info(f"feature_df: {feature_df.shape}")
        logging.info(f"time to preprocess: {(time.time() - start_time)*1000} ms")
        #save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )
        #storage_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"time to save request data: {(time.time() - start_time)*1000} ms")
        # print(feature_df.shape)
        #random [0,1]
        # feature_df_list.append(feature_df)
        # data_id_list.append(data.id)
        # capture_data_path_list.append(self.prob_config.captured_data_dir)
        prediction = self.model.predict(feature_df)
        #is_drifted = self.detect_drift(feature_df)
        logging.info(f"time to predict: {(time.time() - start_time)*1000} ms")
        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": 0,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor_prob1: ModelPredictor,predictor_prob2: ModelPredictor):
        self.predictor1 = predictor_prob1
        self.predictor2 = predictor_prob2
        #self.app = FastAPI()

        @app.get("/")
        async def root():
            return {"message": "hello UIT_DEADLINE"}
        
        @app.post("/phase-1/prob-1/predict")
        async def predict_prob1(data: Data, request: Request):
            #self._log_request(request)
            #logging.info(f"request: {data}")
            response = self.predictor1.predict(data)
            #self._log_response(response)
            return response
        @app.post("/phase-1/prob-2/predict")
        async def predict_prob2(data: Data, request: Request):
            #logging.info(f"request: {data}")
            #self._log_request(request)
            
            response = self.predictor2.predict(data)
            #self._log_response(response)
            return response
        @app.get("/phase-1/get_data")
        async def get_data():
            for i in range(len(feature_df_list)):
                ModelPredictor.save_request_data(feature_df_list[i], capture_data_path_list[i], data_id_list[i])
            feature_df_list.clear()
            data_id_list.clear()
            capture_data_path_list.clear()
            return {"message": "get data success"}
        

    @staticmethod
    def _log_request(request: Request):
        pass
        #logging.info(f"request: {request}")

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        #print(app)
        uvicorn.run(app="model_predictor:app", host="0.0.0.0", port=port,workers=2,log_level="info")

predictor_prob1= ModelPredictor(model_prob1)
predictor_prob2= ModelPredictor(model_prob2)
api = PredictorApi(predictor_prob1,predictor_prob2)

if __name__ == "__main__":
    api.run(PREDICTOR_API_PORT)