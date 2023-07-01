	import argparse
	import logging
	import pickle
	import mlflow
	import numpy as np
	import os
	import xgboost as xgb
	import pandas as pd
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
	from sklearn.model_selection import train_test_split
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
	    df=pd.read_parquet("data/raw_data/phase-1/prob-2/raw_train.parquet")
	    X=df.drop(columns=["label"])
	    y=df["label"]
	    X=RawDataProcessor.apply_category_features(
	            raw_df=X,
	            categorical_cols=prob_config.categorical_cols,
	            category_index=RawDataProcessor.load_category_index(prob_config)
	        )
	    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.5,random_state=np.random.randint(1000))
	    print(train_y)
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
		#get random 10000 sample
		#captured_x=captured_x[np.random.choice(captured_x.shape[0], 10000, replace=False)]
		#captured_y=captured_y[np.random.choice(captured_y.shape[0], 10000, replace=False)]
		index_select=np.random.randint(0, captured_x.shape[0], 10000)
		#captured_x=captured_x[index_select]
		#captured_y=captured_y[index_select]
		captured_x=robust_scaler.transform(captured_x)
		smote = SMOTE(k_neighbors=8,random_state=np.random.randint(1000))
		captured_x, captured_y = smote.fit_resample(captured_x, captured_y)
		train_x = np.concatenate((train_x, captured_x), axis=0)
		train_y = np.concatenate((train_y, captured_y), axis=0)
	    test_x, test_y = RawDataProcessor.load_test_data(prob_config)
	    test_x=robust_scaler.transform(test_x)
	    test_new_x, test_new_y =smote.fit_resample(test_x, test_y)
	    df_test=pd.DataFrame(test_x)
	    df_test["label"]=test_y
	    df_test_new=pd.DataFrame(test_new_x)
	    df_test_new["label"]=test_new_y
	    #get data in test_new but not in test
	    df_test_new=df_test_new[~df_test_new.isin(df_test)].dropna()
	    test_x_new=df_test_new.drop(columns=["label"]).to_numpy()
	    test_y_new=df_test_new["label"].to_numpy()
	    print(train_x.shape)
	    train_x = np.concatenate((train_x, test_x_new), axis=0)
	    train_y = np.concatenate((train_y, test_y_new), axis=0)
	    print(train_x.shape)
	    


	model = ModelTrainer.get_model(model_name)
	model.fit(train_x, train_y)
	predictions = model.predict(test_x)
	auc_scores = roc_auc_score(test_y, predictions)
	f1_scores=f1_score(test_y, predictions)
	precision_scores=precision_score(test_y, predictions)
	recall_scores=recall_score(test_y, predictions)

	metrics = {"test_auc": auc_scores,"f1_score":f1_scores,"precision_score":precision_scores,"recall_score":recall_scores}
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