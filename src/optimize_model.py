import argparse
import logging

import mlflow
import numpy as np
import xgboost as xgb
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import optuna
#from imblearn.over_sampling import SMOTE
from optuna.integration.mlflow import MLflowCallback

def objective(trial,train_x,train_y,valid_x,valid_y):
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }
        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        accuracy =roc_auc_score(valid_y, pred_labels)
        f1=f1_score(valid_y, pred_labels)
        return f1
class ModelOptimize:
    
    EXPERIMENT_NAME=None
    @staticmethod
    def get_model(model_name,objective,model_params):
        dict_model = {
            "xgb": xgb.XGBClassifier(objective=objective, **model_params),
            "svm": SVC(probability=True,C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True),
            "knn": KNeighborsClassifier(),
            "random_forest": RandomForestClassifier(),
            "mlp": MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,100),validation_fraction=0.2,early_stopping=True,solver='adam',learning_rate_init=0.001),
            "ada_boost": AdaBoostClassifier(),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(),

        }
        return  dict_model[model_name]
    @staticmethod
    def optimize_model(prob_config: ProblemConfig, model_params, add_captured_data=False,model_name="xgb"):
        ModelOptimize.EXPERIMENT_NAME = model_name+"-optimze-1"
        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{ModelOptimize.EXPERIMENT_NAME}"
        )

        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        train_x = train_x.to_numpy()
        train_y = train_y.to_numpy()
        #scaler = MinMaxScaler()
        from imblearn.over_sampling import SMOTE
        over_sampling = SMOTE()
        train_x, train_y = over_sampling.fit_resample(train_x, train_y)
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        # if len(np.unique(train_y)) == 2:
        #     objective = "binary:logistic"
        # else:
        #     objective = "multi:softprob"        
        NUM_TRIALS = 20  # small sample for now      
        # Optimize
        mlflow_callback = MLflowCallback(
        tracking_uri=AppConfig.MLFLOW_TRACKING_URI, metric_name=["f1_score"])
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        study = optuna.create_study(study_name=ModelOptimize.EXPERIMENT_NAME, direction="maximize", pruner=pruner)
        study.optimize(lambda trial: objective(trial,train_x,train_y,test_x,test_y), 
                        n_trials=200,
                        callbacks=[mlflow_callback])
        mlflow.log_metric("f1_score", study.best_value)
        mlflow.log_param("best_params", study.best_params)
        # mlflow log
        # mlflow.log_params(model.get_params())
        # mlflow.log_metrics(metrics)
        # signature = infer_signature(test_x, predictions)
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
        #     signature=signature,
        # )
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
    ModelOptimize.optimize_model(
        prob_config, model_config, add_captured_data=args.add_captured_data,model_name=args.model_name
    )
