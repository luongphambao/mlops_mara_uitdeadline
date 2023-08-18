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
from imblearn.under_sampling import RandomUnderSampler
from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig
from sklearn.feature_selection import SelectKBest, mutual_info_classif,chi2
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
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.preprocessing import LabelEncoder
class ModelTrainer:
    
    EXPERIMENT_NAME=None
    @staticmethod
    def get_model(model_name):
        dict_model = {
            "xgb": xgb.XGBClassifier(eval_metric="logloss"),
            "svm": SVC(probability=True,C=100),
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
        test_df=pd.DataFrame()
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
                #smote = SMOTE(sampling_strategy='minority')
                captured_x, captured_y = over_sampling.fit_resample(captured_x, captured_y)
                train_x = np.concatenate((train_x, captured_x), axis=0)
                train_y = np.concatenate((train_y, captured_y), axis=0)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            test_x=robust_scaler.transform(test_x)
            
        if prob_config.phase_id=="phase-2" and prob_config.prob_id=="prob-1":
            #df=pd.read_parquet("data/raw_data/phase-2/prob-1/raw_train.parquet")
            #raw_df=df.copy()
            train_df=pd.read_csv("data/data_phase1_prob1_train_0.csv")
            test_df=pd.read_csv("data/data_phase1_prob1_test_0.csv")
            train_x=train_df.drop(columns=["label"])
            train_y=train_df["label"]
            test_x=test_df.drop(columns=["label"])
            test_y=test_df["label"]
            print(test_x.shape)
            print(train_x.shape)
            #df_add=pd.read_csv("data/predict_test.csv")
            #df_add.drop(columns=["predict","predict_proba"],inplace=True)

            #x_add=df_add.drop(columns=["label","predict","predict_proba"])
            #y_add=df_add["label"]
            #drop data in  df_add in df
            #df=df[~df.index.isin(df_add.index)]

            # X=df.drop(columns=["label"])
            # y=df["label"]
            test_new_x, test_new_y =None,None
            #split train test
            # df_correct=pd.read_csv("data/predict_correct.csv")
            # df_incorrect=pd.read_csv("data/predict_incorrect.csv")
            # correct_x=df_correct.drop(columns=["label","predict","predict_proba"])
            # correct_y=df_correct["label"]
            # incorrect_x=df_incorrect.drop(columns=["label","predict","predict_proba"])
            # incorrect_y=df_incorrect["label"]
            # index_80=int(len(X)*0.6)
            # test_df=raw_df[index_80:]
            # train_x,train_y=X[:index_80],y[:index_80]
            # test_x,test_y=X[index_80:],y[index_80:]
            # print(test_x.shape)
            # print(test_y.shape)
            #add incorrect data
            # train_x=pd.concat([train_x,incorrect_x])
            # train_y=pd.concat([train_y,incorrect_y])
            # print(train_x.shape)
            # print(train_y.shape)
            #print(x_add.shape
            #fill missing category value
            #category_imputer=SimpleImputer(strategy="most_frequent")
            category_imputer=pickle.load(open(os.path.join("features","category_imputer_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"),"rb"))
            train_x[prob_config.categorical_cols]=category_imputer.fit_transform(train_x[prob_config.categorical_cols])
            test_x[prob_config.categorical_cols]=category_imputer.transform(test_x[prob_config.categorical_cols])
            #fill missing numerical value
            #print(train_x.head(20))
            #exit()
            #numerical_imputer=IterativeImputer()
            numerical_imputer=pickle.load(open(os.path.join("features","numerical_imputer_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"),"rb"))
            train_x[prob_config.numerical_cols]=numerical_imputer.fit_transform(train_x[prob_config.numerical_cols])
            test_x[prob_config.numerical_cols]=numerical_imputer.transform(test_x[prob_config.numerical_cols])
            train_x= RawDataProcessor.apply_category_features(
                    raw_df=train_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
            test_x= RawDataProcessor.apply_category_features(
                    raw_df=test_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            train_x=robust_scaler.fit_transform(train_x)
            #name_category_imputer="category_imputer_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            #name_numerical_imputer="numerical_imputer_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            
            name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            # with open(os.path.join("features",name_category_imputer), "wb") as f:
            #     pickle.dump(category_imputer, f)
            # with open(os.path.join("features",name_numerical_imputer), "wb") as f:
            #     pickle.dump(numerical_imputer, f)
            
            with open(os.path.join("features",name_scaler), "wb") as f:
                pickle.dump(robust_scaler, f)
            over_sampling = SMOTE(sampling_strategy="minority",random_state=np.random.randint(1000),k_neighbors=10)
            under_sampling=RandomUnderSampler(random_state=np.random.randint(1000))
            train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            #train_x, train_y = under_sampling.fit_resample(train_x, train_y)
            #test_x,test_y=over_sampling.fit_resample(test_x,test_y)
            #test_x,test_y=under_sampling.fit_resample(test_x,test_y)
            #print(test_x.shape)
            #exit()
            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase2_prob1_model-1.csv"))
                #shuffe data
                #df_capture=df_capture.sample(frac=1)
                #sort by prob in descending
                df_capture=df_capture.sort_values(by=["prob"],ascending=False)
                #print(df_capture.head(20))
                #batch_id,label_model,prob
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                
                captured_x=df_capture.drop(columns=["label_model"])
                captured_x[prob_config.categorical_cols]=category_imputer.transform(captured_x[prob_config.categorical_cols])
                captured_x[prob_config.numerical_cols]=numerical_imputer.transform(captured_x[prob_config.numerical_cols])
                #print(captured_x.head(20))
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                captured_y=df_capture["label_model"]
                captured_x = captured_x.to_numpy()
                captured_y = captured_y.to_numpy()
                #captured_x,test_new_x=captured_x[:4000],captured_x[4000:]
                #captured_y,test_new_y=captured_y[:4000],captured_y[4000:]

                captured_x=robust_scaler.transform(captured_x)
                #test_new_x=robust_scaler.transform(test_new_x)
                smote = SMOTE(sampling_strategy='minority',k_neighbors=10)
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                #test_new_x, test_new_y = smote.fit_resample(test_new_x, test_new_y)
                #test_x = np.concatenate((test_x, captured_x), axis=0)
                #test_y = np.concatenate((test_y, captured_y), axis=0)
                #train_x = np.concatenate((train_x, captured_x), axis=0)
                #train_y = np.concatenate((train_y, captured_y), axis=0)
                train_x=captured_x
                train_y=captured_y
                #test_x=test_new_x
                #test_y=test_new_y
                #test_x = np.concatenate((test_x, test_new_x), axis=0)
                #test_y = np.concatenate((test_y, test_new_y), axis=0)
            #test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            #train_x,train_y=under_sampling.fit_resample(train_x,train_y)
            print(test_x.shape)
            test_x=robust_scaler.transform(test_x)
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
                df_capture=pd.read_csv(os.path.join("data","data_phase2_prob1_model-3.csv"))
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
        #train_x,train_y=under_sampling.fit_resample(train_x,train_y)
        print("final train shape: ",train_x.shape)
        model = ModelTrainer.get_model(model_name)
        model.fit(train_x, train_y)
        predcitions_train=model.predict(train_x)
        print("train accuracy: ",accuracy_score(train_y,predcitions_train))
        predictions = model.predict(test_x)
        predict_proba=model.predict_proba(test_x)
        #find index in test_x predict wrong
        test_df["predict"]=predictions
        test_df["predict_proba"]=np.max(predict_proba,axis=1)
        #find predict!=label
        
        test_df1=test_df[test_df["predict"]!=test_df["label"]]
        test_df1.to_csv("data/predict_incorrect_0.csv",index=False)
        test_df2=test_df[test_df["predict"]==test_df["label"]]
        test_df2.to_csv("data/predict_correct_0.csv",index=False)
        #get confidence
        #confidence=np.max(predict_proba,axis=1)
        
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