import argparse
import logging
import pickle
import mlflow
import numpy as np
import os
import pandas as pd
import xgboost as xgb
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
from features import *
class ModelTrainer:
    
    EXPERIMENT_NAME=None
    @staticmethod
    def get_model(model_name,objective,model_params):
        dict_model = {
            "xgb": xgb.XGBClassifier(objective=objective, **model_params,eval_metric='logloss'),
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
        raw_df=pd.read_parquet(prob_config.raw_data_path)
        X=raw_df.drop(columns=["label"])
        y=raw_df["label"]
        train_x, test_x, train_y, test_y = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=prob_config.random_state,
        )
        imputer = FillUnknown()
        print(train_x)
        feature_name_standardizer = FeatureNamesStandardizer()
        cat_columns =train_x.select_dtypes(include="category").columns.tolist()
        #train_x[cat_columns] =feature_name_standardizer.fit_transform(train_x[cat_columns])
        #test_x[cat_columns] =feature_name_standardizer.transform(test_x[cat_columns])
        train_x = imputer.fit_transform(train_x)
        test_x = imputer.transform(test_x)
        one_hot = PandasOneHot()
        train_x = one_hot.fit_transform(train_x)
        test_x = one_hot.transform(test_x)
        robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
        num_columns = [
                "feature2",
                "feature5",
                "feature13",
                "feature18"
            ]
        train_x[num_columns] = robust_scaler.fit_transform(train_x[num_columns])
        test_x[num_columns] = robust_scaler.transform(test_x[num_columns])
        #save imputer,one_hot,robust_scaler
        with open("features/imputer_prob2.pkl", "wb+") as f:
            pickle.dump(imputer, f)
        with open("features/one_hot_prob2.pkl", "wb+") as f:
            pickle.dump(one_hot, f)
        with open("features/robust_scaler_prob2.pkl", "wb+") as f:
            pickle.dump(robust_scaler, f)
        columns_one_hot_list=train_x.columns.tolist()
        with open("features/columns_one_hot.txt", "w+") as f:
            f.write(str(columns_one_hot_list))
        if len(np.unique(train_y)) == 2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        over_sampling=SMOTE(random_state=prob_config.random_state)
        train_x, train_y = over_sampling.fit_resample(train_x, train_y)
        test_new_x, test_new_y = over_sampling.fit_resample(test_x, test_y)
        df_test=pd.DataFrame(test_x)
        df_test["label"]=test_y
        df_test_new=pd.DataFrame(test_new_x)
        df_test_new["label"]=test_new_y
        #find data in test_new but not in test
        df_test_new=df_test_new[~df_test_new.isin(df_test)].dropna()
        test_new_x=df_test_new.drop(columns=["label"])
        test_new_y=df_test_new["label"]
        # print(train_x.shape)
        # print(test_new_x.shape)
        # print(test_x.head())
        train_x = np.concatenate((train_x, test_new_x), axis=0)
        train_y = np.concatenate((train_y, test_new_y), axis=0)
        test_x=df_test.drop(columns=["label"])
        test_y=df_test["label"]
        if add_captured_data:
                raw_df=pd.read_csv("data/data_phase1_prob2_model-3-4-5-6-7-8.csv")
                raw_df=raw_df.sort_values(by=["prob"],ascending=False)
                label=raw_df["label_model"]
                capture_x=raw_df.drop(columns=["label_model","batch_id","is_drift","prob"])
                capture_y=label
                print("add data")
                
                capture_x=feature_name_standardizer.transform(capture_x)
                capture_x = imputer.transform(capture_x)
                capture_x = one_hot.transform(capture_x)
                capture_x[num_columns] = robust_scaler.transform(capture_x[num_columns])
                print(capture_x)
                print(capture_y.shape)
                #capture_x=robust_scaler.transform(capture_x)
                capture_x, capture_y = over_sampling.fit_resample(capture_x, capture_y)
                train_x = np.concatenate((train_x, capture_x), axis=0)
                train_y = np.concatenate((train_y, capture_y), axis=0)
                print("merge test data")
                
        #using SelectKBest to get top 16 features
        #get_top16_features=SelectKBest(mutual_info_classif, k=16)
        #train_x=get_top16_features.fit_transform(train_x,train_y)
        print("total data: ",len(train_x))
        print(train_x.shape)
        model = ModelTrainer.get_model(model_name,objective,model_params)
        scoring = "recall"
        # kfold = StratifiedKFold(
        #     n_splits=10, shuffle=True, random_state=1
        # )  # Setting number of splits equal to 10
        # cv_result = cross_val_score(
        #     estimator=model, X=train_x, y=train_y, scoring=scoring, cv=kfold )
        model.fit(train_x, train_y)

        # evaluate
        
        #test_x=get_top16_features.transform(test_x)
        #test_x=scaler.transform(test_x)
        #test_x=get_top16_features.transform(test_x)
        print(len(test_x))
        predictions = model.predict(test_x)
        #predict proba >0.8
        predctions_proba=model.predict_proba(test_x)
        #change prediction 1 to 0 and 0 to 1 if proba threshold <0.8
        threshold=0.6
        auc_scores_train = roc_auc_score(train_y, model.predict_proba(train_x)[:,1])

        auc_scores = roc_auc_score(test_y, predictions)
        #predictions[predctions_proba[:,0]<threshold]=1
        #predictions[predctions_proba[:,1]<threshold]=0
        #auc_score_affter=roc_auc_score(test_y,predctions_proba[:,1])
        print("auc score before: ",auc_scores)
        #print("auc score after: ",auc_score_affter)
        f1_scores=f1_score(test_y, predictions)
        precision_scores=precision_score(test_y, predictions)
        recall_scores=recall_score(test_y, predictions)

        metrics = {"train_auc":auc_scores_train,"test_auc": auc_scores,"f1_score":f1_scores,"precision_score":precision_scores,"recall_score":recall_scores}
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
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB2)
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
