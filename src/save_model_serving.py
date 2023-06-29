from typing import Any, Dict, List, Optional
import bentoml
import mlflow
import pandas as pd
from mlflow.models.signature import ModelSignature
from pydantic import BaseModel
import os
import yaml
from utils import AppConfig, AppPath

pd.set_option("display.max_columns", None)



def save_model(config):

    # read from .env file registered_model_version.json, get model name, model version

    model_uri = os.path.join(
            "models:/", config["model_name"], str(config["model_version"])
        )
    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
    mlflow_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    model = mlflow_model._model_impl
    model_signature: ModelSignature = mlflow_model.metadata.signature
    # construct feature list
    feature_list = []
    for name in model_signature.inputs.input_names():
        feature_list.append(name)
    # save model using bentoml
    bentoml_model = bentoml.sklearn.save_model(
        config["model_name"],
        model,
        # model signatures for runner inference
        signatures={
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            },
            "predict_proba": {
                "batchable": True,
                "batch_dim": 0,
            }    
                    },
        labels={
            "owner": "UIT_Deadline",
        },
        metadata={
            "mlflow_model_name": config["model_name"],
            "mlflow_model_version": str(config["model_version"]),
        },
        custom_objects={
            "feature_list": feature_list,
        },
    )
    return bentoml_model

model_prob1="data/model_config/phase-1/prob-1/model-3.yaml"
config_prob1 = yaml.safe_load(open(model_prob1))
model_prob2="data/model_config/phase-1/prob-2/model-3.yaml"
config_prob2 = yaml.safe_load(open(model_prob2))
bentoml_model1 = save_model(config_prob1)
bentoml_model2 = save_model(config_prob2)
print("save model bento service success")

