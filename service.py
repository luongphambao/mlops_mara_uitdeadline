import pandas as pd
from typing import Any, Dict, List, Optional
import bentoml
from bentoml.io import PandasDataFrame,NumpyNdarray,JSON
import yaml 
import os
import sys 
import numpy as np
import asyncio
import json 
from pydantic import BaseModel
import pickle
from pandas.util import hash_pandas_object
from src.problem_config import ProblemConst, create_prob_config
from src.raw_data_processor import RawDataProcessor
from src.utils import AppConfig, AppPath
class MLOPS_SERVING():
    """
    A minimum prediction service exposing a Scikit-learn model
    """
    def __init__(self,model_config_path1,model_config_path2):
        self.config_prob1 = yaml.safe_load(open(model_config_path1))
        self.config_prob2 = yaml.safe_load(open(model_config_path2))
        self.model_prob1_name=self.config_prob1["model_name"]
        self.model_prob2_name=self.config_prob2["model_name"]
        self.model_prob1_version=self.config_prob1["model_version"]
        self.model_prob2_version=self.config_prob2["model_version"]
        self.model_prob1=bentoml.sklearn.get(self.model_prob1_name).to_runner()
        self.model_prob2=bentoml.sklearn.get(self.model_prob2_name).to_runner()
        self.service=bentoml.Service("phase-1",runners=[self.model_prob1,self.model_prob2])
        #self.service_prob2=bentoml.Service(self.model_prob2_name,runners=[self.model_prob2])
        self.prob_config1 = create_prob_config(self.config_prob1["phase_id"], self.config_prob1["prob_id"])
        self.prob_config2 = create_prob_config(self.config_prob2["phase_id"], self.config_prob2["prob_id"])
        self.category_index1 = RawDataProcessor.load_category_index(self.prob_config1)
        self.category_index2 = RawDataProcessor.load_category_index(self.prob_config2)
        #self.columns=self.prob_config1.columns
model_serv=MLOPS_SERVING("data/model_config/phase-1/prob-1/model-3.yaml",
                        "data/model_config/phase-1/prob-2/model-3.yaml")
svc=model_serv.service
#service2=model_serv.service_prob2
model_prob1=model_serv.model_prob1
model_prob2=model_serv.model_prob2
categorical_cols1=model_serv.prob_config1.categorical_cols
categorical_cols2=model_serv.prob_config2.categorical_cols
category_index1=model_serv.category_index1
category_index2=model_serv.category_index2
columns_prob1=["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16"]
#20 features
columns_prob2=["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16","feature17","feature18","feature19","feature20"]
robust_scaler_prob1 = pickle.load(open("features/robust_scaler_phase-1_prob-1.pkl", 'rb'))
robust_scaler_prob2 = pickle.load(open("features/robust_scaler_phase-1_prob-2.pkl", 'rb'))

def save_request_data_json(request,captured_data_dir):
    """Save request data to json file"""
    result_path=os.path.join(captured_data_dir, f"{request.id}.json")
    dict_request=request.dict()
    with open(result_path,"w") as f:
        json.dump(dict_request,f)
def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
    if data_id.strip():
        filename = data_id
    else:
        filename = hash_pandas_object(feature_df).sum()
    output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
    feature_df.to_parquet(output_file_path, index=False)
    return output_file_path
# def save_respone_data(feature5: list, captured_data_dir,label,data_id):
#     """Save response data and featuredf"""
#     #create df 
respone_phase1_prob1_dir="data/captured_data/phase-1/prob-1/response"
respone_phase1_prob2_dir="data/captured_data/phase-1/prob-2/response"
if not os.path.exists(respone_phase1_prob1_dir):
    os.makedirs(respone_phase1_prob1_dir)
if not os.path.exists(respone_phase1_prob2_dir):
    os.makedirs(respone_phase1_prob2_dir)
def save_respone_data_phase1_prob1(feature_df,predictions,data_id):
    """Save response data and featuredf to json file"""
    feature_df["label_model"]=predictions
    feature_df.to_parquet(os.path.join(respone_phase1_prob1_dir,data_id+".parquet"),index=False)
def save_respone_data_phase1_prob2(feature_df,predictions,data_id):
    """Save response data and featuredf to json file"""
    feature_df["label_model"]=predictions
    feature_df.to_parquet(os.path.join(respone_phase1_prob2_dir,data_id+".parquet"),index=False)

class InferenceRequest(BaseModel):
    id: str
    rows: List[list]
    columns: List[str]
class InferenceResponse(BaseModel):
    id: Optional[str]
    predictions: Optional[List[float]]
    drift: Optional[int]
def predict(request: np.ndarray) -> np.ndarray:
    result = model_prob1.predict.run(request)
    return result
def post_process(result,confidence_score,threshold):
    # change class 1 to 0 if result=1 and confidence_score<threshold
    dict_change={1:0}
    result=[0 if result[i]==1 and confidence_score[i]<threshold else result[i] for i in range(len(result))]
    return result
@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route ="/phase-1/prob-1/predict",
   
)
async def inference_phase1_prob1(request: InferenceRequest):
    """
    Example request: {"id": "123", "rows": [[113, 9, 63.17, 44420, 43.08059354989101, -82.53012356265731, 15429, 40.282905, -80.067555, 24.265891758842134, 20, 0, 5, 0.8992553559883496], [-1, 5, 6.7, 84015, 40.951731452865886, -112.89138115985779, 62494, 40.565544, -112.367142, 19.10470490940412, 23, 5, 7, 0.7518914565676378]], "columns": ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16"]}
    """
    save_request_data_json(request, model_serv.prob_config1.captured_data_dir)
    rows=request.rows
    raw_df = pd.DataFrame(rows, columns=request.columns)
    feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=categorical_cols1,
            category_index=category_index1)[columns_prob1]
    feature_df=robust_scaler_prob1.transform(feature_df)
    result = await asyncio.gather(model_prob1.predict.async_run(feature_df))
    result=result[0]
    result=[int(i) for i in result]
    feature_df=pd.DataFrame(feature_df,columns=columns_prob1)
    response = InferenceResponse()
    response.id=request.id
    response.predictions=result
    response.drift=0
    #save_respone_data_phase1_prob1(feature_df,result,request.id)
    return response
@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route ="/phase-1/prob-2/predict")
async def inference_phase1_prob2(request: InferenceRequest):
    """example request:
        {
    "id": "123",
    "rows": [
      [
        2,
        40.951731452865886,
        6.7,
        84015,
        40.951731452865886,
        -112.89138115985779,
        62494,
        40.565544,
        -112.367142,
        19.10470490940412,
        23,
        5,
        7.23224434,
        0.7518914565676378,
        1.3916673053951893,
        76.89848528466531,
        7,
        0.7518914565676378,
        1.3916673053951893,
        76.89848528466531
      ]
    ],
    "columns": [
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
      "feature20"
    
    ]
  }
  """
    
    save_request_data_json(request, model_serv.prob_config2.captured_data_dir)
    rows=request.rows
    raw_df = pd.DataFrame(rows, columns=request.columns)
    feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=categorical_cols2,
            category_index=category_index2)
    feature_df=feature_df[columns_prob2]
    feature_df=robust_scaler_prob2.transform(feature_df)
    feature_df=pd.DataFrame(feature_df,columns=columns_prob2)
    result = await asyncio.gather(model_prob2.predict.async_run(feature_df))
    result=result[0]
    result=[int(i) for i in result]
    #batch_id=str(batch_id)
   # print("batch_id",batch_id)
    response = InferenceResponse()
    response.id=request.id
    response.predictions=result
    response.drift=0
    #save_respone_data_phase1_prob2(feature_df,result,request.id)
    return response
