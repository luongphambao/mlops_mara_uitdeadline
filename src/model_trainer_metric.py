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
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
)
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
    def get_model(model_name,objective,model_params):
        dict_model = {
            "xgb": xgb.XGBClassifier(objective=objective, **model_params,eval_metric="logloss"),
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

        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        train_x = train_x.to_numpy()
        train_y = train_y.to_numpy()
        
        #RobustScaler for last 14 features
        robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
        train_x=robust_scaler.fit_transform(train_x)
        #save scaler
        name_scaler="robust_scaler_"+str(prob_config.phase_id)+"_"+str(prob_config.prob_id)+".pkl"
        with open(os.path.join("features",name_scaler), "wb+") as f:
            pickle.dump(robust_scaler, f)
        
        over_sampling = SMOTE(sampling_strategy="minority")
        train_x, train_y = over_sampling.fit_resample(train_x, train_y)
        # train model
        if len(np.unique(train_y)) == 2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        
        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            captured_x = captured_x.to_numpy()
            captured_y = captured_y.to_numpy()
            #print
            #merge captured data_x and data_y
            captured = np.concatenate((captured_x, captured_y.reshape(-1, 1)), axis=1)
            #drop duplicates
            captured = np.unique(captured, axis=0)
            #print(len(captured))
            captured_x = captured[:, :-1]
            captured_y = captured[:, -1]
            captured_x=robust_scaler.transform(captured_x)
            train_x = np.concatenate((train_x, captured_x), axis=0)
            train_y = np.concatenate((train_y, captured_y), axis=0)
            #captured_x=robust_scaler.transform(captured_x)
            # predictions_uncertain = model.predict_proba(captured_x)
            # precitions=model.predict(captured_x)
            # #get list prob of predicted class
            # #prob_predicted_class = predictions_uncertain[np.arange(len(predictions_uncertain)), predictions_uncertain.argmax(1)]
            # import pandas as pd
            # #save df with X,y,prediction,confidence
            # columns_name=["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16","uncertain_y","y_predict","confidence"]
            # confidence = np.max(predictions_uncertain, axis=1)
            # data_save=np.concatenate((captured_x,captured_y.reshape(-1,1),precitions.reshape(-1,1),confidence.reshape(-1,1)),axis=1)

            # df_save=pd.DataFrame(data_save,columns=columns_name)
            # df_save.to_csv("report_prob1.csv",index=False)

            # auc_score=roc_auc_score(captured_y, predictions_uncertain[:,1])
            # print(auc_score)
        model = ModelTrainer.get_model(model_name,objective,model_params)
        model.fit(train_x, train_y)
        # evaluate
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        test_x=robust_scaler.transform(test_x)
        #test_x=scaler.transform(test_x)
        #test_x=get_top16_features.transform(test_x)
        predictions = model.predict(test_x)
        predictions_uncertain = model.predict_proba(test_x)
        confidence = np.max(predictions_uncertain, axis=1)
        #save auc with confidence>0.9

        auc_scores = roc_auc_score(test_y, predictions)
        f1_scores=f1_score(test_y, predictions)
        precision_scores=precision_score(test_y, predictions)
        recall_scores=recall_score(test_y, predictions)
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
