import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from problem_config import ProblemConfig, ProblemConst, get_prob_config
import os 
import json
import pickle 
try:
    from problem_config import ProblemConst, create_prob_config
    from raw_data_processor import RawDataProcessor
    from utils import AppConfig, AppPath
except:
    from src.problem_config import ProblemConst, create_prob_config
    from src.raw_data_processor import RawDataProcessor
    from src.utils import AppConfig, AppPath
def label_captured_data(prob_config: ProblemConfig):
    #train_x = pd.read_parquet(prob_config.train_x_path).to_numpy()
    #train_y = pd.read_parquet(prob_config.train_y_path).to_numpy()
    #test_x = pd.read_parquet(prob_config.test_x_path).to_numpy()
    #test_y = pd.read_parquet(prob_config.test_y_path).to_numpy()
    train_df=pd.read_csv("data/predict_incorrect.csv")
    #drop predict,predict_proba
    train_df=train_df.drop(["predict","predict_proba"],axis=1)
    train_x=train_df.drop(["label"],axis=1)
    train_y=train_df["label"]
    train_y=train_y.to_numpy()
    category_impute_imputer=pickle.load(open("features/category_imputer_phase-2_prob-1.pkl","rb"))
    numeric_imputer=pickle.load(open("features/numerical_imputer_phase-2_prob-1.pkl","rb"))
    train_x[prob_config.categorical_cols]=category_impute_imputer.transform(train_x[prob_config.categorical_cols])
    train_x[prob_config.numerical_cols]=numeric_imputer.transform(train_x[prob_config.numerical_cols])
    train_x=RawDataProcessor.apply_category_features(
            raw_df=train_x,
            categorical_cols=prob_config.categorical_cols,
            category_index=RawDataProcessor.load_category_index(prob_config),
        )
    

    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    print(len(os.listdir(prob_config.captured_data_dir)))
    # for file_path in prob_config.captured_data_dir.glob("*.json"):
    #     captured_data =json.load(open(file_path))
    #     rows = captured_data["rows"]
    #     columns = captured_data["columns"]
    #     captured_data = pd.DataFrame(rows, columns=columns)
    #     captured_x = pd.concat([captured_x, captured_data])
    # #reindex feature1 to feature41
    # captured_x = captured_x.reindex(columns=["feature" + str(i) for i in range(1, 42)])
    if prob_config.prob_id == "prob-1":
        #captured_x=pd.read_csv("data/data_captured_phase-2_prob-1.csv")
        captured_x=pd.read_csv("data/data_phase2_prob1_model-1.csv")
        
        df_pre=captured_x[:8000].copy()
        captured_x=captured_x.drop(["label_model","prob"],axis=1)
        y_pre=df_pre["label_model"].to_numpy()
        print(df_pre)
        captured_x=captured_x[8000:]
        print(captured_x.head())
        #exit()
        #drop label_model,prob
        

    else:
        captured_x=pd.read_csv("data/data_phase2_prob1_model-1.csv")
        captured_x=captured_x[8000:]
        #drop label_model,prob
        
    #drop duplicates
    df_raw=captured_x.copy()
    print(captured_x.shape)
    #captured_x = captured_x.drop_duplicates()
    #drop nan
    #captured_x = captured_x.dropna()
    category_index = RawDataProcessor.load_category_index(prob_config)
    captured_x = RawDataProcessor.apply_category_features(
            raw_df=captured_x,
            categorical_cols=prob_config.categorical_cols,
            category_index=category_index,
        )
    np_captured_x = captured_x.to_numpy()
    #print(train_x)
    
    train_x = train_x.to_numpy()
    print(train_x)
    n_captured = len(np_captured_x)
    n_samples = len(train_x) + n_captured
    logging.info(f"Loaded {n_captured} captured samples, {n_samples} train + captured")

    logging.info("Initialize and fit the clustering model")
    #n_cluster = int(n_samples /5) * len(np.unique(train_y))
    n_cluster =500
    print(n_cluster)
    kmeans_model = MiniBatchKMeans(
        n_clusters=n_cluster, random_state=prob_config.random_state
    ).fit(train_x)

    logging.info("Predict the cluster assignments for the new data")
    kmeans_clusters = kmeans_model.predict(np_captured_x)

    logging.info(
        "Assign new labels to the new data based on the labels of the original data in each cluster"
    )
    new_labels = []
    for i in range(n_cluster):
        mask = kmeans_model.labels_ == i  # mask for data points in cluster i
        cluster_labels = train_y[mask]  # labels of data points in cluster i
        if len(cluster_labels) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(0)
        else:
            # For a linear regression problem, use the mean of the labels as the new label
            # For a logistic regression problem, use the mode of the labels as the new label
            if ml_type == "regression":
                new_labels.append(np.mean(cluster_labels.flatten()))
            else:
                new_labels.append(
                    np.bincount(cluster_labels.flatten()).argmax()
                )

    approx_label = [new_labels[c] for c in kmeans_clusters]
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])
    df_raw["label_model"]=approx_label
    df_raw["prob"]=[0.8]*len(df_raw)
    #merge with df_pre
    df_raw=pd.concat([df_pre,df_raw],ignore_index=True)
    df_raw.to_csv("data/data_captured_phase-2_prob-1_kmean2.csv",index=False)
    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    label_captured_data(prob_config)
