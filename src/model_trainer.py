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
#train test split
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE,RandomOverSampler,KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler

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
#import simple imputer and iterrative imputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import json 
#encoder label
from sklearn.preprocessing import LabelEncoder
#import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
class ModelTrainer:
    
    EXPERIMENT_NAME=None
    @staticmethod
    def get_model(model_name):
        dict_model = {
            #"xgb": xgb.XGBClassifier(max_depth=10,learning_rate=0.01,sampling_method="gradient_based",n_estimators=1000,objective="binary:logistic",booster="gbtree",random_state=42),
            "xgb":xgb.XGBClassifier(random_state=np.random.randint(0,1000),n_estimators=200,max_depth=7),
            "svm": SVC(probability=True,C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True),
            "knn": KNeighborsClassifier(n_neighbors=20),
            "random_forest": RandomForestClassifier(),
            "mlp": MLPClassifier(),
            "ada_boost": AdaBoostClassifier(),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(),
            "lightgbm": LGBMClassifier(lambda_l1=9.10851272776884e-05,lambda_l2=0.00030844070554030494,num_leaves=212,feature_fraction=0.6617910556625971,bagging_fraction=0.982605608542239,bagging_freq=5,min_child_samples=10),

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
            columns=train_x.columns
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
            #robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            robust_scaler=pickle.load(open(os.path.join("features","robust_scaler_phase-1_prob-1.pkl"),"rb"))
            train_x=robust_scaler.fit_transform(train_x)
            # name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            # with open(os.path.join("features",name_scaler), "wb") as f:
            #     pickle.dump(robust_scaler, f)
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
                smote =KMeansSMOTE(k_neighbors=5,sampling_strategy='minority')
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                train_x = np.concatenate((train_x, captured_x), axis=0)
                train_y = np.concatenate((train_y, captured_y), axis=0)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            test_x=robust_scaler.transform(test_x)
        if prob_config.phase_id=="phase-2" and prob_config.prob_id=="prob-1":
            #df=pd.read_parquet("data/raw_data/phase-2/prob-1/raw_train.parquet")
            df=pd.read_csv("data/raw_train.csv")
            df=df.sort_values(by=['prob'],ascending=False)
            df=df.drop(columns=["prob","label_model"])
            X=df.drop(columns=["label"])
            y=df["label"]
            index_train= 50000
            train_x,test_x=X[ :index_train],X[index_train:]
            train_y,test_y=y[ :index_train],y[index_train:]
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
            #get 20000 sample shuffe
            train_x,train_y=train_x,train_y
            over_sampling = SMOTE()
            under_sampling=RandomUnderSampler()
            #train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            
            over_sampling_test=SMOTE(k_neighbors=10)
            test_new_x,test_new_y=over_sampling_test.fit_resample(test_x, test_y)
            #print 
            #drop test_x in test_x_sample and test_y in test_y_sample
            test_new_x,test_new_y=test_new_x[len(test_x):],test_new_y[len(test_y):]

            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase2_prob1_model-1.csv"))
                #add 5000 sample by duplicate
                df_add=pd.read_csv(os.path.join("data","data_phase2_prob1_model-1.csv"))
                # #random 5000 sample
                # df_add=df_add.sample(n=10000)
                df_add1=pd.read_csv(os.path.join("data","data_phase2_prob1_model-1.csv"))
                #get 5000 sample sort by increasing prob
                df_add=df_add.sort_values(by=['prob'],ascending=False)[:5000]
                df_add1=df_add1.sort_values(by=['prob'],ascending=True)[:5000]

                df_capture=pd.concat([df_capture,df_add,df_add1],axis=0)

                df_capture=df_capture.sort_values(by=['prob'],ascending=True)
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                if "label" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["label"])
                #print(df_capture.head())
                captured_x=df_capture.drop(columns=["label_model"])
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                captured_y=df_capture["label_model"]
                print(len(captured_x))
                #captured_x = captured_x.to_numpy()[:7000]
                #captured_y = captured_y.to_numpy()[:7000]
                index=np.random.randint(int(len(captured_x)*0.5),len(captured_x))
                captured_x,test_new_x=captured_x[:index],captured_x[index:]
                captured_y,test_new_y=captured_y[:index],captured_y[index:]
                #add 100-500 sample test_new_x,test_new_y to captured_x,captured_y
                
                #captured_x=robust_scaler.transform(captured_x)
                #test_new_x=robust_scaler.transform(test_new_x)
                #over_samling_capture=RandomOverSampler(sampling_strategy='all')

                smote = SMOTE(k_neighbors=10)
                under_sampling=RandomUnderSampler()

                
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                test_x=np.concatenate((test_x,train_x),axis=0)
                test_y=np.concatenate((test_y,train_y),axis=0)
                train_x=captured_x
                train_y=captured_y
                print(test_new_x.shape)

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
        if prob_config.phase_id=="phase-3" and prob_config.prob_id=="prob-1":
            #df=pd.read_csv("data/raw_train.csv")
            # df=df.sort_values(by=['prob'],ascending=False)
            # df=df.drop(columns=["prob","label_model"])
            #drop duplicate
            df=pd.read_parquet("data/raw_data/phase-2/prob-1/raw_train.parquet")
            df=df.drop_duplicates()
            X=df.drop(columns=["label"])
            y=df["label"]
            columns=df.columns
            add_x=X
            add_y=y
            add_x=RawDataProcessor.apply_category_features(
                    raw_df=add_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )            
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            train_x=pd.concat([train_x,add_x],axis=0)
            train_y=pd.concat([train_y,add_y],axis=0)
            #drop duplicates
            #train_x=train_x.drop_duplicates()
            #train_y=train_y.drop_duplicates()
            
            #train_x = train_x.to_numpy()
            #train_y = train_y.to_numpy()
            # robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            # train_x=robust_scaler.fit_transform(train_x)
            # name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            # with open(os.path.join("features",name_scaler), "wb") as f:
            #     pickle.dump(robust_scaler, f)
            # df_train=pd.DataFrame(train_x)
            # df_train["label"]=train_y
            # df_train.to_csv("train1.csv",columns=columns,index=False)
            robust_scaler = pickle.load(open(os.path.join("features","robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"), 'rb'))
            train_x=robust_scaler.transform(train_x)
            over_sampling = SMOTE()
            under_sampling=RandomUnderSampler()
            train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            train_x, train_y = under_sampling.fit_resample(train_x, train_y)
            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase3_prob1_model-1.csv"))
                
                #shuffe data
                df_capture=df_capture.sample(frac=1)
                #sort prob
                #df_capture=df_capture.sort_values(by=['prob'],ascending=False)
                #batch_id,label_model,prob
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                
                captured_x=df_capture.drop(columns=["label_model"])
                captured_y=df_capture["label_model"]
                #print(captured_y
                #captured_x=captured_x.to_numpy()
                #captured_y=captured_y.to_numpy()
                num_index=int(len(captured_x)*0.7)
                
                #print(captured_x)
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                # df_capture=pd.DataFrame(captured_x,columns=captured_x.columns)
                # df_capture["label_model"]=captured_y
                # df_capture.to_csv("capture.csv",index=False)
                captured_x,test_new_x=captured_x[:num_index],captured_x[num_index:]
                captured_y,test_new_y=captured_y[:num_index],captured_y[num_index:]

                captured_x=robust_scaler.transform(captured_x)
                test_new_x=robust_scaler.transform(test_new_x)
                smote = SMOTE(sampling_strategy='minority')
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                #train_x = np.concatenate((train_x, captured_x), axis=0)
                #train_y = np.concatenate((train_y, captured_y), axis=0)
                train_x=np.concatenate((train_x, captured_x), axis=0)
                train_y=np.concatenate((train_y, captured_y), axis=0)
                # test_x=np.concatenate((test_x, test_new_x), axis=0)
                # test_y=np.concatenate((test_y, test_new_y), axis=0)

            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            #test_y=encoder_label.transform(test_y)
            # df_test=pd.DataFrame(test_x)
            # df_test["label"]=test_y
            
            # df_test.to_csv("test.csv",columns=columns,index=False)
            test_x=robust_scaler.transform(test_x)
            #merge train_x and train_y
            
            
        if prob_config.phase_id=="phase-3" and prob_config.prob_id=="prob-2":
            add_df=pd.read_parquet("data/raw_data/phase-2/prob-2/raw_train.parquet")
            #drop duplicate
            add_df=add_df.drop_duplicates()
            add_x=add_df.drop(columns=["label"])
            add_y=add_df["label"]
            add_x=RawDataProcessor.apply_category_features(
                    raw_df=add_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
            add_x_train, add_x_test, add_y_train, add_y_test = train_test_split(add_x, add_y, test_size=0.3, random_state=42)
            
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            #train_x=pd.concat([train_x,add_x],axis=0)
            #train_y=pd.concat([train_y,add_y],axis=0)

            encoder_label=LabelEncoder()
            train_y=encoder_label.fit_transform(train_y)
            add_y_train=encoder_label.transform(add_y_train)
            add_y_test=encoder_label.transform(add_y_test)
            #get mapping label
            mapping_label={}
            for i in range(len(encoder_label.classes_)):
                mapping_label[i]=encoder_label.classes_[i]
            with open(os.path.join("features","mapping_label_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".json"), "w+") as f:
                json.dump(mapping_label, f)
            train_x = train_x.to_numpy()
            add_x_train = add_x_train.to_numpy()
            add_x_test = add_x_test.to_numpy()
            #train_y = train_y.to_numpy()
            robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
            train_x=robust_scaler.fit_transform(train_x)
            add_x_train=robust_scaler.transform(add_x_train)
            add_x_test=robust_scaler.transform(add_x_test)
            
            name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
            with open(os.path.join("features",name_scaler), "wb") as f:
                pickle.dump(robust_scaler, f)
            over_sampling = SMOTE()
            train_x, train_y = over_sampling.fit_resample(train_x, train_y)
            under_sampling=RandomUnderSampler()
            #train_x, train_y = under_sampling.fit_resample(train_x, train_y)
            if add_captured_data:
                df_capture=pd.read_csv(os.path.join("data","data_phase3_prob2_model-1.csv"))
                #shuffe data
                #df_capture=df_capture.sample(frac=1)
                #sort by prob
                df_capture=df_capture.sort_values(by=['prob'],ascending=False)

                #batch_id,label_model,prob
                if "batch_id" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["batch_id"])
                if "is_drift" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["is_drift"])
                if "prob" in df_capture.columns:
                    df_capture=df_capture.drop(columns=["prob"])
                
                captured_x=df_capture.drop(columns=["label_model"])
                captured_y=df_capture["label_model"]
                captured_x=RawDataProcessor.apply_category_features(
                    raw_df=captured_x,
                    categorical_cols=prob_config.categorical_cols,
                    category_index=RawDataProcessor.load_category_index(prob_config)
                )
                captured_x=captured_x.to_numpy()
                captured_y=captured_y.to_numpy()
                num_index=int(len(captured_x)*0.95)
                captured_x,test_new_x=captured_x[:num_index],captured_x[num_index:]
                captured_y,test_new_y=captured_y[:num_index],captured_y[num_index:]
                
                
                captured_y=encoder_label.transform(captured_y)
                #captured_x = captured_x.to_numpy()
                #captured_y = captured_y.to_numpy()
                captured_x=robust_scaler.transform(captured_x)
                smote = SMOTE(sampling_strategy='minority')
                captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
                train_x = np.concatenate((train_x, captured_x), axis=0)
                train_y = np.concatenate((train_y, captured_y), axis=0)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            test_y=encoder_label.transform(test_y)
            test_x=robust_scaler.transform(test_x)
            train_x,train_y=np.concatenate((train_x, add_x_train), axis=0),np.concatenate((train_y, add_y_train), axis=0)
            test_x,test_y=np.concatenate((test_x, add_x_test), axis=0),np.concatenate((test_y, add_y_test), axis=0)
        print("train_x shape: ",train_x.shape)
        print("test_x shape: ",test_x.shape)
        model = ModelTrainer.get_model(model_name)
        model.fit(train_x, train_y)
        predictions_train = model.predict(train_x)
        print("train accuracy: ",accuracy_score(train_y, predictions_train))
        predictions = model.predict(test_x)
        
        if prob_config.phase_id=="phase-3" and prob_config.prob_id=="prob-2":
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
            metrics = {"f1_score":f1_scores,"precision_score":precision_scores,"recall_score":recall_scores,"accuracy_score":accuracy_scores,"train_accuracy":accuracy_score(train_y, predictions_train),"auc_score":auc_scores}
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