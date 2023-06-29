import argparse
import logging
import pickle
import mlflow
import numpy as np
import os
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
import pyarrow.parquet as pq
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from feature import FeatureNamesStandardizer,FillUnknown,PandasOneHot
def auc_custom(confidence_min,confidence_max,y_test,y_pred,confidence):
    """
    get auc with confidence threshold
    input: 
        confidence_min: min confidence
        confidence_max: max confidence
        y_test: y_test
        y_pred: y_pred
        confidence: confidence
    output:
        auc_score_confidence: auc score with confidence threshold
    """
    y_test=y_test.to_numpy()
    y_pred_confidence=y_pred[(confidence_min<=confidence) & (confidence<=confidence_max)]
    y_test_confidence=y_test[(confidence_min<=confidence) & (confidence<=confidence_max)]
    auc_score_confidence=roc_auc_score(y_test_confidence, y_pred_confidence)
    f1_score_confidence=f1_score(y_test_confidence, y_pred_confidence)
    # print(y_test_confidence)
    # print(y_pred_confidence)
    # print(auc_score_confidence)
    # print(f1_score_confidence)
    #count predict correct percent per class
    count_correct_class_0=0
    count_correct_class_1=0
    count_class_0=0
    count_class_1=0
    for i in range(len(y_pred_confidence)):
        if y_test_confidence[i]==0:
            count_class_0+=1
            if y_pred_confidence[i]==0:
                count_correct_class_0+=1
        else:
            count_class_1+=1
            if y_pred_confidence[i]==1:
                count_correct_class_1+=1
    percent_0=count_correct_class_0/count_class_0
    percent_1=count_correct_class_1/count_class_1
    return auc_score_confidence,f1_score_confidence,percent_0,percent_1
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
        #process raw data
        #get random seed
        seed=np.random.randint(0,1000)
        df=pq.read_table(prob_config.raw_data_path).to_pandas()
        X,y = df.drop(columns=["label"]),  df["label"]
        
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=prob_config.test_size, random_state=seed)
        #train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        # train_x = train_x.to_numpy()
        # train_y = train_y.to_numpy()
        print(train_x["feature18"].value_counts())
        # feature_name_standardizer = FeatureNamesStandardizer()
        # train_x = feature_name_standardizer.fit_transform(train_x)
        # test_x = feature_name_standardizer.transform(test_x)
        cat_columns = train_x.select_dtypes(include="category").columns.tolist()
        #imputer=FillUnknown()
        obj_columns_names = df.select_dtypes(include=['object']).columns.to_list()
        cols = ['feature8', 'feature11', 'feature16', 'feature18'] + obj_columns_names
        print(cols)
        encoder = ce.TargetEncoder(cols=cols)

        #train_x[cat_columns] = imputer.fit_transform(train_x[cat_columns])
        #test_x[cat_columns] = imputer.transform(test_x[cat_columns])
        train_x = encoder.fit_transform(train_x, train_y)
        test_x = encoder.transform(test_x)
        with open(os.path.join("features","target_encoder_prob2.pkl"), "wb") as f:
            pickle.dump(encoder, f)
        num_columns = ['feature2', 'feature5', 'feature13']
        #RobustScaler for last 14 features
        robust_scaler = RobustScaler(with_centering=False,with_scaling=True)
        train_x[num_columns] = robust_scaler.fit_transform(train_x[num_columns])

        test_x[num_columns] = robust_scaler.transform(test_x[num_columns])
        #save scaler
        name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
        with open(os.path.join("features",name_scaler), "wb") as f:
            pickle.dump(robust_scaler, f)
    
        over_sampling = SMOTE(sampling_strategy="minority", random_state=seed)
        over_sampling2 = SMOTE()
        train_x, train_y = over_sampling.fit_resample(train_x, train_y)
        test_new_x, test_new_y = over_sampling2.fit_resample(test_x, test_y)
        df_test=pd.DataFrame(test_x,columns=test_x.columns)
        df_test["label"]=test_y
        df_test_new=pd.DataFrame(test_new_x,columns=test_new_x.columns)
        df_test_new["label"]=test_new_y
        #find data in test_new but not in test
        df_test_new=df_test_new[~df_test_new.isin(df_test)].dropna()
        test_new_x=df_test_new.drop(columns=["label"])
        test_new_y=df_test_new["label"]
        train_x=np.concatenate((train_x,test_new_x),axis=0)
        train_y=np.concatenate((train_y,test_new_y),axis=0)
        # train model
  
        
        print("prepare data done")
        print(train_x.shape)
        model = ModelTrainer.get_model(model_name)
        model.fit(train_x, train_y)
        # evaluate
        # test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        # test_x=robust_scaler.transform(test_x)
        # #test_x=scaler.transform(test_x)
        # #test_x=get_top16_features.transform(test_x)
        
        predictions = model.predict(test_x)
        auc_scores = roc_auc_score(test_y, predictions)
        f1_scores=f1_score(test_y, predictions)
        precision_scores=precision_score(test_y, predictions)
        recall_scores=recall_score(test_y, predictions)
        predictions_uncertain = model.predict_proba(test_x)
        confidence = np.max(predictions_uncertain, axis=1)
        auc_score_09,f1_score_09,percent_0_09,percent_1_09=auc_custom(0.9,1,test_y,predictions,confidence)
        auc_score_08,f1_score_08,percent_0_08,percent_1_08=auc_custom(0.8,0.9,test_y,predictions,confidence)
        auc_score_07,f1_score_07,percent_0_07,percent_1_07=auc_custom(0.7,0.8,test_y,predictions,confidence)
        metrics = {"test_auc": auc_scores,"f1_score":f1_scores,"precision_score":precision_scores,"recall_score":recall_scores,"auc_score_09":auc_score_09,"f1_score_09":f1_score_09,"percent_0_09":percent_0_09,"percent_1_09":percent_1_09,"auc_score_08":auc_score_08,"f1_score_08":f1_score_08,"percent_0_08":percent_0_08,"percent_1_08":percent_1_08,"auc_score_07":auc_score_07,"f1_score_07":f1_score_07,"percent_0_07":percent_0_07,"percent_1_07":percent_1_07}
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
