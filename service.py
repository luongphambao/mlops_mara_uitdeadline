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
from src.drift_predict import DriftPredictor


LOG_PATH = os.environ.get("/sample_solution/data/monitoring")

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
        self.robust_scaler_prob1 =bentoml.sklearn.get("scaler_phase-3_prob-1").to_runner()
        self.robust_scaler_prob2 =bentoml.sklearn.get("scaler_phase-3_prob-2").to_runner()
        self.service=bentoml.Service("phase-3",runners=[self.model_prob1,self.model_prob2,self.robust_scaler_prob1,self.robust_scaler_prob2])
        #self.service_prob2=bentoml.Service(self.model_prob2_name,runners=[self.model_prob2])
        self.prob_config1 = create_prob_config(self.config_prob1["phase_id"], self.config_prob1["prob_id"])
        self.prob_config2 = create_prob_config(self.config_prob2["phase_id"], self.config_prob2["prob_id"])
        self.category_index1 = RawDataProcessor.load_category_index(self.prob_config1)
        self.category_index2 = RawDataProcessor.load_category_index(self.prob_config2)
        #self.columns=self.prob_config1.columns
model_serv=MLOPS_SERVING("data/model_config/phase-3/prob-1/model-1.yaml",
                        "data/model_config/phase-3/prob-2/model-1.yaml")
svc=model_serv.service
#service2=model_serv.service_prob2
model_prob1=model_serv.model_prob1
model_prob2=model_serv.model_prob2
categorical_cols1=model_serv.prob_config1.categorical_cols
categorical_cols2=model_serv.prob_config2.categorical_cols
numerical_cols1=model_serv.prob_config1.numerical_cols
numerical_cols2=model_serv.prob_config2.numerical_cols
category_index1=model_serv.category_index1
category_index2=model_serv.category_index2
#41 features
columns_prob1=["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16","feature17","feature18","feature19","feature20","feature21","feature22","feature23","feature24","feature25","feature26","feature27","feature28","feature29","feature30","feature31","feature32","feature33","feature34","feature35","feature36","feature37","feature38","feature39","feature40","feature41"]
#41 features
columns_prob2=["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16","feature17","feature18","feature19","feature20","feature21","feature22","feature23","feature24","feature25","feature26","feature27","feature28","feature29","feature30","feature31","feature32","feature33","feature34","feature35","feature36","feature37","feature38","feature39","feature40","feature41"]
mapping_phase2_prob2=json.load(open("features/mapping_label_phase-3_prob-2.json","r"))
robust_scaler_prob1=model_serv.robust_scaler_prob1
robust_scaler_prob2=model_serv.robust_scaler_prob2
#robust_scaler_prob1 = pickle.load(open("features/robust_scaler_phase-3_prob-1.pkl", 'rb'))
#robust_scaler_prob2 = pickle.load(open("features/robust_scaler_phase-3_prob-2.pkl", 'rb'))
#robust_scaler_prob1 =bentoml.sklearn.get("scaler_phase-3_prob-1").to_runner()
#robust_scaler_prob2 =bentoml.sklearn.get("scaler_phase-3_prob-2").to_runner()
DriftPredictor_prob1=DriftPredictor(phase_id="phase-3",prob_id="prob-1")
DriftPredictor_prob2=DriftPredictor(phase_id="phase-3",prob_id="prob-2")
respone_phase3_prob1_dir="data/captured_data/phase-3/prob-1/response"
respone_phase3_prob2_dir="data/captured_data/phase-3/prob-2/response"
#category_imputer_prob1 = pickle.load(open("features/category_imputer_phase-2_prob-1.pkl", 'rb'))
#category_imputer_prob2 = pickle.load(open("features/category_imputer_prob2.pkl", 'rb'))
numerical_imputer_prob1 = pickle.load(open("features/numerical_imputer_phase-2_prob-1.pkl", 'rb'))
if not os.path.exists(respone_phase3_prob1_dir):
    os.makedirs(respone_phase3_prob1_dir)
if not os.path.exists(respone_phase3_prob2_dir):
    os.makedirs(respone_phase3_prob2_dir)
request_phase2_prob1_list=[]
request_phase2_prob2_list=[]
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

# def save_respone_data_phase1_prob1(feature_df,predictions,data_id):
#     """Save response data and featuredf to json file"""
#     feature_df["label_model"]=predictions
#     feature_df.to_parquet(os.path.join(respone_phase1_prob1_dir,data_id+".parquet"),index=False)
# def save_respone_data_phase1_prob2(feature_df,predictions,data_id):
#     """Save response data and featuredf to json file"""
#     feature_df["label_model"]=predictions
#     feature_df.to_parquet(os.path.join(respone_phase1_prob2_dir,data_id+".parquet"),index=False)

class InferenceRequest(BaseModel):
    id: str
    rows: List[list]
    columns: List[str]
class InferenceResponse(BaseModel):
    id: Optional[str]
    predictions: Optional[List[float]]
    drift: Optional[int]
class InferenceDataCollection(BaseModel):
    phase_id: str
    prob_id: str
class InferenceStatus(BaseModel):
    status: str
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
    #request_phase1_prob1_list.append(request)
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
    #feature_df=pd.DataFrame(feature_df,columns=columns_prob1)
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
    #request_phase1_prob2_list.append(request)
    #print(request_phase1_prob2_list)
    #save_request_data_json(request, model_serv.prob_config2.captured_data_dir)
    rows=request.rows
    raw_df = pd.DataFrame(rows, columns=request.columns)
    feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=categorical_cols2,
            category_index=category_index2)
    feature_df=feature_df[columns_prob2]
    feature_df=robust_scaler_prob2.transform(feature_df)
    #feature_df=pd.DataFrame(feature_df,columns=columns_prob2)
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
@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route ="/phase-2/prob-1/predict")
async def inference_phase2_prob1(request: InferenceRequest):
    save_request_data_json(request, model_serv.prob_config1.captured_data_dir)
    raw_df = pd.DataFrame(request.rows, columns=request.columns)
    #print(raw_df.shape)
    raw_df=raw_df[columns_prob1]
    #print(raw_df.head())
    #raw_df[categorical_cols1]=category_imputer_prob1.transform(raw_df[categorical_cols1])
    #raw_df[numerical_cols1]=numerical_imputer_prob1.transform(raw_df[numerical_cols1])
    #raw_df=numerical_imputer_prob1.transform(raw_df)
    feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=categorical_cols1,
            category_index=category_index1)
    feature_df=feature_df[columns_prob1]
    #feature_df=robust_scaler_prob1.transform(feature_df)
    result = await asyncio.gather(model_prob1.predict.async_run(feature_df))
    result=result[0]
    result=[int(i) for i in result]

    response = InferenceResponse()
    response.id=request.id
    response.predictions=result
    response.drift=0
    return response
@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route ="/phase-2/prob-2/predict")
async def inference_phase2_prob2(request: InferenceRequest):
    save_request_data_json(request, model_serv.prob_config2.captured_data_dir)
    raw_df = pd.DataFrame(request.rows, columns=request.columns)
    print(raw_df.shape)
    feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=categorical_cols2,
            category_index=category_index2)
    feature_df=feature_df[columns_prob2]
    feature_df=robust_scaler_prob2.transform(feature_df)
    result = await asyncio.gather(model_prob2.predict.async_run(feature_df))
    result=result[0]
    result=[int(i) for i in result]
    result=[mapping_phase2_prob2[str(i)] for i in result]
    response = InferenceResponse()
    response.id=request.id
    response.predictions=result
    response.drift=0
    return response
@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route ="/phase-3/prob-1/predict")
async def inference_phase3_prob1(request: InferenceRequest):
    #save_request_data_json(request, model_serv.prob_config1.captured_data_dir)
    raw_df = pd.DataFrame(request.rows, columns=request.columns)
    #print(raw_df.shape)
    raw_df=raw_df[columns_prob1]
    is_drift=DriftPredictor_prob1.predict_data_drift(raw_df)
    #print(raw_df.head())
    #raw_df[categorical_cols1]=category_imputer_prob1.transform(raw_df[categorical_cols1])
    #raw_df[numerical_cols1]=numerical_imputer_prob1.transform(raw_df[numerical_cols1])
    #raw_df=numerical_imputer_prob1.transform(raw_df)
    feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=categorical_cols1,
            category_index=category_index1)
    feature_df=feature_df[columns_prob1]
    #print(feature_df.head())
    feature_df=await asyncio.gather(robust_scaler_prob1.transform.async_run(feature_df))
    feature_df=feature_df[0]
    result = await asyncio.gather(model_prob1.predict.async_run(feature_df))
    result=result[0]
    result=[int(i) for i in result]

    response = InferenceResponse()
    response.id=request.id
    response.predictions=result
    response.drift=is_drift
    return response
@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route ="/phase-3/prob-2/predict")
async def inference_phase3_prob2(request: InferenceRequest):
    #save_request_data_json(request, model_serv.prob_config2.captured_data_dir)
    raw_df = pd.DataFrame(request.rows, columns=request.columns)
    #print(raw_df.shape)
    raw_df=raw_df[columns_prob2]
    is_drift=DriftPredictor_prob2.predict_data_drift(raw_df)
    feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=categorical_cols2,
            category_index=category_index2)
    feature_df=feature_df[columns_prob2]
    feature_df=await asyncio.gather(robust_scaler_prob2.transform.async_run(feature_df))
    feature_df=feature_df[0]
    result = await asyncio.gather(model_prob2.predict.async_run(feature_df))
    result=result[0]
    result=[int(i) for i in result]
    result=[mapping_phase2_prob2[str(i)] for i in result]
    response = InferenceResponse()
    response.id=request.id
    response.predictions=result
    response.drift=is_drift
    return response


# @svc.api(
#     input=JSON(pydantic_model=InferenceDataCollection),
#     output=JSON(pydantic_model=InferenceStatus),
#     route ="/phase-1/prob-1/data",)
# def data_collection_phase1_prob1(request: InferenceDataCollection):
#     """example request:
#         {
#     "phase_id": "phase-1",
#     "prob_id": "prob-1"
#   }
#   """
#     for request in request_phase1_prob1_list:
#         save_request_data_json(request, model_serv.prob_config1.captured_data_dir)
#     request_phase1_prob1_list.clear()
#     return InferenceStatus(status="storage phase-1 prob-1 data successfully")
# @svc.api(
#     input=JSON(pydantic_model=InferenceDataCollection),
#     output=JSON(pydantic_model=InferenceStatus),
#     route ="/phase-1/prob-2/data",)
# def data_collection_phase1_prob2(request: InferenceDataCollection):
#     """example request:
#         {
#     "phase_id": "phase-1",
#     "prob_id": "prob-2"
#   }
#   """
#     for request in request_phase1_prob2_list:
#         save_request_data_json(request, model_serv.prob_config2.captured_data_dir)
#     request_phase1_prob2_list.clear()
#     return InferenceStatus(status="storage phase-1 prob-2 data successfully")

# @svc.on_shutdown
# def save_request_phase_1_prob1():
#     """Save request data to json file"""
#     for request in request_phase1_prob1_list:
#         save_request_data_json(request, model_serv.prob_config1.captured_data_dir)
#     #request_phase1_prob1_list.clear()
# def save_request_phase_1_prob2():
#     """Save request data to json file"""
#     for request in request_phase1_prob2_list:
#         save_request_data_json(request, model_serv.prob_config2.captured_data_dir)
#     #request_phase1_prob2_list.clear()
    
