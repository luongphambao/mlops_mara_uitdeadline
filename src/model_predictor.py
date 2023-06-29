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


class Settings(BaseSettings):
    # ... The rest of our FastAPI settings

    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"
    

settings = Settings()


def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass
USE_NGROK = True 

PREDICTOR_API_PORT = 8000
app = FastAPI()
model_prob1="data/model_config/phase-1/prob-1/model-2.yaml"
model_prob2="data/model_config/phase-1/prob-2/model-1.yaml"


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
        #time.sleep(0.02)
        return random.choice([0, 1])

    def predict(self, data: Data):
        start_time = time.time()

        # preprocess
        #print(str(self.prob_config.captured_data_dir)+"/"+str(data.id)+".csv")
        try:
            raw_df = pd.DataFrame(data.rows, columns=data.columns)
            
            raw_df.to_csv(str(self.prob_config.captured_data_dir)+"/"+str(data.id)+".csv",index=False)
            feature_df = RawDataProcessor.apply_category_features(
                raw_df=raw_df,
                categorical_cols=self.prob_config.categorical_cols,
                category_index=self.category_index,
            )
            # save request data for improving models
            # ModelPredictor.save_request_data(
            #     feature_df, self.prob_config.captured_data_dir, data.id
            # )
            print(feature_df.shape)
            #random [0,1]
            prediction = self.model.predict(feature_df)
        except:
            #log exception
            print("error")
            prediction=[0 for i in range(len(data.rows))]
            return {
                "id": data.id,
                "predictions": prediction,
                "drift": 0,
            }
        is_drifted = 0

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        #feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor_prob1: ModelPredictor,predictor_prob2: ModelPredictor):
        self.predictor1 = predictor_prob1
        self.predictor2 = predictor_prob2
        #self.app = FastAPI()

        @app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        @app.get("/")
        async def root():
            return {"message": "hello UIT_DEADLINE"}

        @app.post("/phase-1/prob-1/predict")
        async def predict_prob1(data: Data, request: Request):
            self._log_request(request)
            #logging.info(f"request: {data}")
            response = self.predictor1.predict(data)
            self._log_response(response)
            return response
        @app.post("/phase-1/prob-2/predict")
        async def predict_prob2(data: Data, request: Request):
            #logging.info(f"request: {data}")
            self._log_request(request)
            response = self.predictor2.predict(data)
            self._log_response(response)
            return response
    
        

    @staticmethod
    def _log_request(request: Request):
        pass
        #logging.info(f"request: {request}")

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        print(app)
        uvicorn.run(app="model_predictor:app", host="0.0.0.0", port=port,workers=2)

predictor_prob1= ModelPredictor(model_prob1)
predictor_prob2= ModelPredictor(model_prob2)
api = PredictorApi(predictor_prob1,predictor_prob2)
if __name__ == "__main__":
    api.run(PREDICTOR_API_PORT)