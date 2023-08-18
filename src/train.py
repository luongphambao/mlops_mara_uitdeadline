import argparse
import logging
import pickle
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

import json
#encoder label
from sklearn.preprocessing import LabelEncoder
#import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# data_path="data/raw_data/phase-3/prob-1/raw_train.parquet"
# data_path1="data/raw_data/phase-2/prob-1/raw_train.parquet"
# data=pd.read_parquet(data_path)
# #drop duplicate
# data.drop_duplicates(inplace=True)
# data1=pd.read_parquet(data_path1)
# data1.drop_duplicates(inplace=True)
# data=pd.concat([data,data1],axis=0)
# categorical_cols=['feature2','feature3','feature4']
# data,category_index=RawDataProcessor.build_category_features(data,categorical_cols)
# print(data.head())
# #exit()
# X=data.drop(columns=['label'])
# y=data['label']
# #encoding categorical feature (feature2,feature3,feature4)
# le=LabelEncoder()



# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.random.randint(0,100))
# X_train=X_train.to_numpy()
# X_test=X_test.to_numpy()
df_train=pd.read_csv("train_data.csv")
df_test=pd.read_csv("test_data.csv")
df_val=pd.read_csv("data/data_phase3_prob1_model-1.csv")
#sort by decreasing prob
df_val=df_val.sort_values(by=['prob'],ascending=True)
df_val_get=df_val.copy()
categorical_cols=['feature2','feature3','feature4']
#merge train and test
#df_train=pd.concat([df_train,df_test],axis=0)
df_train,category_index=RawDataProcessor.build_category_features(df_train,categorical_cols)
df_test=RawDataProcessor.apply_category_features(df_test,categorical_cols,category_index)
df_val=RawDataProcessor.apply_category_features(df_val,categorical_cols,category_index)
scaler=RobustScaler(with_centering=True,with_scaling=True)
X_val=df_val.drop(columns=['label_model'])
y_val=df_val['label_model']
#drop prob column
X_val.drop(columns=['prob'],inplace=True)

X_train=df_train.drop(columns=['label'])
y_train=df_train['label']
print(y_train.value_counts())
X_test=df_test.drop(columns=['label'])
y_test=df_test['label']
scaler=RobustScaler(with_centering=False,with_scaling=True)
# X_train=scaler.fit_transform(X_train)
# #X_train=scaler.fit_transform(X_train)
# X_test=scaler.transform(X_test)
# X_val=scaler.transform(X_val)
num_index=len(X_val)*0.4
num_index=int(num_index)

#get random 50 sample from val
auc_tests=[]
auc_trains=[]
index_vals=[]
for i in range(100):
    X_val1,X_add=X_val[:num_index],X_val[num_index:]
    y_val1,y_add=y_val[:num_index],y_val[num_index:]
    index_val=np.random.randint(0,len(X_val1),100)
    index_vals.append(index_val)
    X_val1=X_val1.iloc[index_val]
    y_val1=y_val1.iloc[index_val]
    X_val1=scaler.fit_transform(X_val1)
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    #smote=SMOTE(random_state=42)
    #X_train,y_train=smote.fit_resample(X_train,y_train)
    model=LGBMClassifier(max_depth=10,n_estimators=1000,random_state=np.random.randint(0,100))

    model.fit(X_val1,y_val1)
    y_pred=model.predict(X_test)
    y_pred_proba=model.predict_proba(X_test)
    auc_test=roc_auc_score(y_test,y_pred)
    auc_tests.append(auc_test)

    print("-----------------test set-----------------")
    print("roc_auc_score: ",roc_auc_score(y_test,y_pred))
    # print("f1_score: ",f1_score(y_test,y_pred))
    # print("precision_score: ",precision_score(y_test,y_pred))
    # print("recall_score: ",recall_score(y_test,y_pred))
    print("accuracy_score: ",accuracy_score(y_test,y_pred))
    #train set
    print("-----------------train set-----------------")
    y_pred=model.predict(X_train)
    y_pred_proba=model.predict_proba(X_train)
    print("roc_auc_score: ",roc_auc_score(y_train,y_pred))
    auc_train=roc_auc_score(y_train,y_pred)
    auc_trains.append(auc_train)
    # print("f1_score: ",f1_score(y_train,y_pred))
    # print("precision_score: ",precision_score(y_train,y_pred))
    # print("recall_score: ",recall_score(y_train,y_pred))
    print("accuracy_score: ",accuracy_score(y_train,y_pred))
print("auc_test: ",np.mean(auc_tests))
print("auc_train: ",np.mean(auc_trains))
print("best auc_test: ",np.max(auc_tests))
print("best auc_train: ",np.max(auc_trains))
print("index_val: ",index_vals[np.argmax(auc_tests)])
index_val_best=index_vals[np.argmax(auc_tests)]
X_val_best=X_val.iloc[index_val_best]
y_val_best=y_val.iloc[index_val_best]
df_val_best=df_val_get.iloc[index_val_best]
csv_path="correct_data1.csv"
if os.path.exists(csv_path):
    data=pd.read_csv(csv_path)
    data=pd.concat([data,df_val_best],axis=0)
    #drop duplicate
    data.drop_duplicates(inplace=True)
    data.to_csv(csv_path,index=False)
else:
    df_val_best.to_csv(csv_path,index=False)
    