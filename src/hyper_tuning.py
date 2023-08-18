"""
Optuna example that demonstrates a pruner for LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.

You can run this example as follows:
    $ python lightgbm_integration.py

"""
import numpy as np
import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    df_train=pd.read_csv("train1.csv")
    df_test=pd.read_csv("test.csv")
    df_add=pd.read_csv("capture.csv")
    #remove prob column
    #df_add=df_add.drop(["prob"],axis=1)
    #rename label_model to label
    df_add=df_add.rename(columns={"label_model":"label"})
    df_train=pd.concat([df_train,df_add],axis=0)
    #remo
    train_x=df_train.drop(["label"],axis=1)
    train_y=df_train["label"]
    valid_x=df_test.drop(["label"],axis=1)
    valid_y=df_test["label"]
    scaler=pickle.load(open("features/robust_scaler_phase-3_prob-1.pkl","rb"))
    train_x=scaler.transform(train_x)
    valid_x=scaler.transform(valid_x)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)

    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "n_estimators": 200,
        #"colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        #"min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30), direction="maximize"
    )
    study.optimize(objective, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    #print
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

# Best trial:
#   Value: 0.9503244863198292
#   Params:
#     lambda_l1: 9.10851272776884e-05
#     lambda_l2: 0.00030844070554030494
#     num_leaves: 212
#     feature_fraction: 0.6617910556625971
#     bagging_fraction: 0.982605608542239
#     bagging_freq: 5
#     min_child_samples: 10