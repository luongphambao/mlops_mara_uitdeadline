import pandas as pd
import os 
from pycaret.classification import *
mlflow_env="/app/mlops_mara_uitdeadline/deployment/mlflow/run_env/data"
model_path="mlartifacts/17/03e1697c74fa4675a0a2e264145de7b0/artifacts/model/model"
model=load_model(os.path.join(mlflow_env,model_path))
data=pd.read_parquet("../data/raw_data/phase-2/prob-2/raw_train.parquet")
data=data.drop(columns=["label"])
result=predict_model(model,data)
prediction_label=result["prediction_label"]
prediction_score=result["prediction_score"]
print(prediction_label)