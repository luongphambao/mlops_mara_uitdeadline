import argparse
import logging
import pickle
import mlflow
import numpy as np
import os
import xgboost as xgb
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
)
import json 
#encoder label
from sklearn.preprocessing import LabelEncoder
class ModelTrainer:
    
    EXPERIMENT_NAME=None
    @staticmethod
    def get_model(model_name):
        dict_model = {
            "xgb": xgb.XGBClassifier(eval_metric="logloss"),
            "svm": SVC(probability=True,C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True),
            "knn": KNeighborsClassifier(),
            "random_forest": RandomForestClassifier(),
            "mlp": MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,100),validation_fraction=0.2,early_stopping=True,solver='adam',learning_rate_init=0.001),
            "ada_boost": AdaBoostClassifier(),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(),
            "lightgbm": LGBMClassifier(),

        }
        return  dict_model[model_name]
    @staticmethod
    def train_model(prob_config: ProblemConfig, model_params, add_captured_data=False,model_name="xgb"):
        ModelTrainer.EXPERIMENT_NAME = model_name+"-1"
        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{ModelTrainer.EXPERIMENT_NAME}"
        )
        if prob_config.phase_id=="phase-1" and prob_config.prob_id=="prob-1":
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            train_x=robust_scaler.fit_transform(train_x)
            name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            with open(os.path.join("features",name_scaler), "wb") as f:
                pickle.dump(robust_scaler, f)
            over_sampling = SMOTE()
            train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            
            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase1_prob1_model-3.csv"))
                #batch_id,label_model,prob
                
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                captured_x=df_capture.drop(columns=["label_model"])
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                captured_y=df_capture["label_model"]
                captured_x = captured_x.to_numpy()
                captured_y = captured_y.to_numpy()
                captured_x=robust_scaler.transform(captured_x)
                smote = SMOTE(sampling_strategy='minority')
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                train_x = np.concatenate((train_x, captured_x), axis=0)
                train_y = np.concatenate((train_y, captured_y), axis=0)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            test_x=robust_scaler.transform(test_x)
        if prob_config.phase_id=="phase-1" and prob_config.prob_id=="prob-2":
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            train_x=robust_scaler.fit_transform(train_x)
            name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            with open(os.path.join("features",name_scaler), "wb") as f:
                pickle.dump(robust_scaler, f)
            over_sampling = SMOTE()
            train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase1_prob2_model-3.csv"))
                #batch_id,label_model,prob
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                
                captured_x=df_capture.drop(columns=["label_model"])
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                captured_y=df_capture["label_model"]
                captured_x = captured_x.to_numpy()
                captured_y = captured_y.to_numpy()
                captured_x=robust_scaler.transform(captured_x)
                smote = SMOTE(sampling_strategy='minority')
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                train_x = np.concatenate((train_x, captured_x), axis=0)
                train_y = np.concatenate((train_y, captured_y), axis=0)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            test_x=robust_scaler.transform(test_x)
        if prob_config.phase_id=="phase-2" and prob_config.prob_id=="prob-1":
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            print(train_y.value_counts())
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            train_x=robust_scaler.fit_transform(train_x)
            name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            with open(os.path.join("features",name_scaler), "wb") as f:
                pickle.dump(robust_scaler, f)
            over_sampling = SMOTE()
            train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase2_prob1_model-1.csv"))
                RawDataProcessor.fill_category_features(df_capture,prob_config.categorical_cols)
                RawDataProcessor.fill_numeric_features(df_capture,prob_config.numerical_cols)
                
                #shuffe data
                #df_capture=df_capture.sample(frac=1)
                #sort by descending prob
                df_capture=df_capture.sort_values(by=['prob'],ascending=False)
                df_capture_last=df_capture.iloc[8000:]
                df_capture_last.to_csv("data/last.csv",index=False)
                #batch_id,label_model,prob
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                
                captured_x=df_capture.drop(columns=["label_model"])
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                captured_y=df_capture["label_model"]
                print(len(captured_x))
                captured_x = captured_x.to_numpy()
                captured_y = captured_y.to_numpy()
                captured_x,test_new_x=captured_x[:8000],captured_x[8000:]
                captured_y,test_new_y=captured_y[:8000],captured_y[8000:]

                captured_x=robust_scaler.transform(captured_x)
                test_new_x=robust_scaler.transform(test_new_x)
                smote = SMOTE(sampling_strategy='minority')
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                #test_new_x,test_new_y=smote.fit_resample(test_new_x, test_new_y)
                train_x = np.concatenate((train_x, captured_x), axis=0)
                train_y = np.concatenate((train_y, captured_y), axis=0)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            test_x=robust_scaler.transform(test_x)
            #concat train_x and test_x 
            test_x,test_y= over_sampling.fit_resample(test_x, test_y)
            train_x=np.concatenate((train_x,test_x),axis=0)
            train_y=np.concatenate((train_y,test_y),axis=0)
            test_x=test_new_x
            test_y=test_new_y

        if prob_config.phase_id=="phase-2" and prob_config.prob_id=="prob-2":
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            encoder_label=LabelEncoder()
            train_y=encoder_label.fit_transform(train_y)
            #get mapping label
            mapping_label={}
            for i in range(len(encoder_label.classes_)):
                mapping_label[i]=encoder_label.classes_[i]
            with open(os.path.join("features","mapping_label_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".json"), "w+") as f:
                json.dump(mapping_label, f)
            train_x = train_x.to_numpy()
            #train_y = train_y.to_numpy()
            robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            train_x=robust_scaler.fit_transform(train_x)
            name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            with open(os.path.join("features",name_scaler), "wb") as f:
                pickle.dump(robust_scaler, f)
            over_sampling = SMOTE()
            train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase2_prob2_model-1.csv"))
                #shuffe data
                df_capture=df_capture.sample(frac=1)
                #batch_id,label_model,prob
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                
                captured_x=df_capture.drop(columns=["label_model"])
                captured_y=df_capture["label_model"]
                captured_x=captured_x.to_numpy()
                captured_y=captured_y.to_numpy()
                captured_x,test_new_x=captured_x[:5000],captured_x[5000:]
                captured_y,test_new_y=captured_y[:5000],captured_y[5000:]
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                
                captured_y=encoder_label.transform(captured_y)
                captured_x = captured_x.to_numpy()
                #captured_y = captured_y.to_numpy()
                captured_x=robust_scaler.transform(captured_x)
                smote = SMOTE(sampling_strategy='minority')
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                train_x = np.concatenate((train_x, captured_x), axis=0)
                train_y = np.concatenate((train_y, captured_y), axis=0)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            test_y=encoder_label.transform(test_y)
            test_x=robust_scaler.transform(test_x) 
        print("train_x shape: ",train_x.shape)
        print("test_x shape: ",test_x.shape)
        model = ModelTrainer.get_model(model_name)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        
        # accuracy_scores=accuracy_score(test_y, predictions)
        # f1_scores=f1_score(test_y, predictions)
        # precision_scores=precision_score(test_y, predictions)
        # recall_scores=recall_score(test_y, predictions)
        if prob_config.phase_id=="phase-2" and prob_config.prob_id=="prob-2":
            accuracy_scores=accuracy_score(test_y, predictions)
            f1_scores=f1_score(test_y, predictions,average="micro")
            precision_scores=precision_score(test_y, predictions,average="micro")
            recall_scores=recall_score(test_y, predictions,average="micro")
            metrics = {"f1_score":f1_scores,"precision_score":precision_scores,"recall_score":recall_scores,"accuracy_score":accuracy_scores}
        else:
            accuracy_scores=accuracy_score(test_y, predictions)
            f1_scores=f1_score(test_y, predictions)
            precision_scores=precision_score(test_y, predictions)
            recall_scores=recall_score(test_y, predictions)
            auc_scores = roc_auc_score(test_y, predictions)
            metrics = {"auc": auc_scores,"f1_score":f1_scores,"precision_score":precision_scores,"recall_score":recall_scores,"accuracy_score":accuracy_scores}
        logging.info(f"metrics: {metrics}")

        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
        )
        mlflow.end_run()
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("--model-name", type=str, default="xgb")
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    model_config = {"random_state": prob_config.random_state}
    ModelTrainer.train_model(
        prob_config, model_config, add_captured_data=args.add_captured_data,model_name=args.model_name
    )